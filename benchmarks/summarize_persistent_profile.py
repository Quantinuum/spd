"""Summarize GPU kernel durations from the bounded AFH profiler runs."""
import json
from collections import defaultdict
from pathlib import Path
import sys


def summarize(path):
    result=json.loads(path.read_text())
    trace=json.loads(Path(result['profile']['trace']).read_text())
    groups=defaultdict(lambda: {'calls':0,'seconds':0.})
    grids=defaultdict(lambda: {'calls':0,'seconds':0.})
    streams=set()
    for event in trace['traceEvents']:
        if event.get('ph')!='X' or event.get('cat')!='kernel':
            continue
        name=event['name']
        group=next((k for k in ('make_planes','select_rows','update_variant','insert_variant','compact_variant') if k in name),'other kernels')
        groups[group]['calls']+=1
        groups[group]['seconds']+=event['dur']/1e6
        streams.add(event.get('args',{}).get('stream'))
        if group=='make_planes':
            grid=str(event.get('args',{}).get('grid'))
            grids[grid]['calls']+=1
            grids[grid]['seconds']+=event['dur']/1e6
    return dict(variant=result['args']['variant'],unprofiled_seconds=result['steps'][0]['seconds'],
        instrumented_seconds=result['profile']['instrumented_seconds'],
        work=result['steps'][0]['work'],rows=result['profile']['rows'],
        energy=result['profile']['energy'],norm2=result['profile']['norm2'],
        topk_calls=sum(e.get('cat')=='cpu_op' and e.get('name')=='aten::topk' for e in trace['traceEvents']),
        kernel_seconds=sum(g['seconds'] for g in groups.values()),streams=sorted(streams),
        groups=dict(groups),plane_grids=dict(sorted(grids.items(),key=lambda p:-p[1]['seconds'])))

if __name__=='__main__':
    for arg in sys.argv[1:]:
        path=Path(arg)
        result=summarize(path)
        path.with_suffix('.kernel_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='plane_grids'},indent=2))
