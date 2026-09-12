"""Instrument the unchanged stepwise benchmark outside its timed regions.

Select allocator policy before Python starts using PYTORCH_CUDA_ALLOC_CONF.
Records live storage, allocator counters, and final segment occupancy. Cache
release is tested only after evolution finishes; it never enters a gate timer.
"""

import argparse
from collections import Counter
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


def counters():
    stats = torch.cuda.memory_stats()
    names = [
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "requested_bytes.all.current",
        "requested_bytes.all.peak",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "inactive_split_bytes.all.current",
        "inactive_split_bytes.all.peak",
        "segment.all.current",
        "segment.all.allocated",
        "num_alloc_retries",
        "num_ooms",
        "num_device_alloc",
        "num_device_free",
    ]
    return {name: stats.get(name) for name in names}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-steps", type=int, default=14)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/benchmark_2d_obc_xx_z_stepwise.py"
    spec = importlib.util.spec_from_file_location("allocator_stepwise", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    result = dict(
        allocator_config=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        torch=torch.__version__,
        triton=__import__("triton").__version__,
        device=torch.cuda.get_device_name(),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        steps=[],
    )
    out = args.output_dir / "allocator.json"
    latest_storage = {}
    original_evolve = example.evolve_step
    original_save = example.save_benchmark_data

    def evolve(*args, **kwargs):
        state, elapsed = original_evolve(*args, **kwargs)
        latest_storage.update(
            live_rows=state.get_size(),
            logical_bytes=sum(
                t.numel() * t.element_size() for t in [state.xz_array, state.c_array]
            ),
            backing_storage_bytes=sum(
                t.untyped_storage().nbytes() for t in [state.xz_array, state.c_array]
            ),
        )
        return state, elapsed

    def save(path, data):
        original_save(path, data)
        if len(data["times"]) == len(result["steps"]):
            return
        torch.cuda.synchronize()
        record = dict(
            step=len(data["times"]),
            seconds=data["times"][-1],
            **latest_storage,
            counters=counters(),
        )
        result["steps"].append(record)
        if len(result["steps"]) == args.num_steps:
            segments = torch.cuda.memory_snapshot()
            result["segment_summary"] = dict(
                segments=len(segments),
                fully_inactive_bytes=sum(
                    s["total_size"] for s in segments if s["active_size"] == 0
                ),
                inactive_bytes_in_active_segments=sum(
                    s["total_size"] - s["active_size"]
                    for s in segments
                    if s["active_size"] != 0
                ),
                block_states=dict(
                    Counter(b["state"] for s in segments for b in s["blocks"])
                ),
            )
            # Aggregated occupancy is enough to distinguish caching from fragmentation.
            result["segments"] = [
                {
                    k: s[k]
                    for k in [
                        "total_size",
                        "allocated_size",
                        "active_size",
                        "segment_type",
                    ]
                }
                for s in segments
            ]
        out.write_text(json.dumps(result, indent=2) + "\n")
        print("allocator:", json.dumps(record), flush=True)

    example.evolve_step = evolve
    example.save_benchmark_data = save
    sys.argv = [
        str(path),
        "--backend",
        "triton",
        "--num-steps",
        str(args.num_steps),
        "--output-dir",
        str(args.output_dir),
    ]
    example.main()
    gc.collect()
    torch.cuda.synchronize()
    result["after_state_release"] = counters()
    torch.cuda.empty_cache()
    result["after_final_empty_cache"] = counters()
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
