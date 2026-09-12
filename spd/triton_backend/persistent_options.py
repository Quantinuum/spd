"""Independent, opt-in experiments layered on the committed persistent baseline."""
from functools import lru_cache
import math
import torch
from . import persistent as base
from .experiment_kernels import (insert_variant, update_variant, compact_variant,
                                  make_planes, select_rows, begin_device,
                                  insert_device, advance_device)

MODES = ('baseline', 'control', 'local', 'inverted', 'select', 'soa', 'fingerprint', 'buffers', 'device', 'inverted+local', 'inverted+fingerprint', 'inverted+local+fingerprint', 'select+local', 'local_all', 'select_all', 'inverted_all', 'select_all+local_all', 'inverted_all+local_all')
_BaseStorage = base._Storage


@lru_cache(maxsize=4096)
def descriptor(pauli, device):
    sites = [(q, p) for q, p in enumerate(pauli) if p != 'I']
    kind = 1 if len(sites) == 1 and sites[0][1] == 'Z' else 2 if len(sites) == 2 and all(p == 'X' for q,p in sites) else 0
    q = sites[0][0] if sites else 0
    r = sites[-1][0] if sites else 0
    return kind, torch.tensor([q//32, 31-q%32, r//32, 31-r%32], dtype=torch.int32, device=device)


@lru_cache(maxsize=4096)
def local_descriptor(pauli, device):
    """Word, bit, X flag and Z flag for each of at most two occupied sites."""
    occupied = [(q, p) for q, p in enumerate(pauli) if p != 'I']
    if not 1 <= len(occupied) <= 2:
        return 0, None
    values = [v for q, p in occupied for v in (q//32, 31-q%32, int(p in 'XY'), int(p in 'YZ'))]
    return len(occupied), torch.tensor(values, dtype=torch.int32, device=device)


class OptionStorage(_BaseStorage):
    def __init__(self, state, dead_fraction, mode, descriptors, local_descriptors=None):
        self.mode, self.descriptors = mode, descriptors
        self.local_descriptors = {} if local_descriptors is None else local_descriptors
        self.features = set(mode.split('+'))
        self.inverted = bool(self.features.intersection(('inverted', 'inverted_all')))
        self.fingerprints = self.planes = self.selected = None
        self.pool = None
        self.plane_capacity = 0
        self.first = True
        super().__init__(state, dead_fraction)

    def key_args(self, keys=None):
        keys = self.keys if keys is None else keys
        return dict(STRIDE=keys.stride(1), SOA=keys.stride(0) == 1 and 'soa' in self.features)

    def new_keys(self, capacity, final=False):
        if 'soa' in self.features and not final:
            return torch.empty((self.width, capacity), dtype=torch.int32, device=self.device).T
        return torch.empty((capacity, self.width), dtype=torch.int32, device=self.device)

    def allocate_pair(self, capacity, final=False):
        if 'buffers' in self.features and not final and self.pool is not None and self.pool[0].shape[0] >= capacity:
            keys, coeff = self.pool
            self.pool = None
            return keys, coeff
        return self.new_keys(capacity, final), torch.empty(capacity, dtype=self.dtype, device=self.device)

    def recycle_pair(self, keys, coeff):
        if 'buffers' in self.features and (self.pool is None or keys.shape[0] > self.pool[0].shape[0]):
            self.pool = keys, coeff

    def _planes(self, start=0):
        needed = (self.n + 31)//32
        if needed > self.plane_capacity:
            capacity = max(1, (needed*5+3)//4)
            planes = torch.empty((self.width*32, capacity), dtype=torch.int32, device=self.device)
            if start and self.planes is not None:
                planes[:, :self.plane_capacity].copy_(self.planes)
            self.planes, self.plane_capacity = planes, capacity
        if self.n:
            make_planes[((self.n+31)//32-start//32, self.width)](
                self.keys, self.planes, start, self.n, self.plane_capacity, self.width)

    def _insert_range(self, start):
        if start < self.n:
            insert_variant[((self.n-start+127)//128,)](
                self.keys, self.table, start, self.n, self.table_size-1, self.width, 128,
                **self.key_args(), fingerprints=self.fingerprints, FP='fingerprint' in self.features)

    def _rebuild(self):
        if 'soa' in self.features and self.keys.stride(0) != 1:
            keys = self.new_keys(self.capacity)
            keys[:self.n].copy_(self.keys[:self.n])
            self.keys = keys
        size = 1 << (max(4, 4*self.n)-1).bit_length()
        if 'buffers' in self.features and hasattr(self, 'table') and self.table.numel() == size:
            self.table.fill_(-1)
        else:
            self.table = torch.full((size,), -1, dtype=torch.int32, device=self.device)
        self.table_size = size
        if 'fingerprint' in self.features:
            self.fingerprints = torch.empty(self.capacity, dtype=torch.int32, device=self.device)
        self._insert_range(0)
        if self.inverted:
            self._planes()
        self.rebuilds += 1

    def _reserve(self, required=None):
        needed = 2*self.n if required is None else required
        if needed <= self.capacity:
            return
        capacity = max(needed, int(self.capacity*1.25))
        keys, coeff = self.allocate_pair(capacity)
        capacity = keys.shape[0]
        keys[:self.n].copy_(self.keys[:self.n])
        coeff[:self.n].copy_(self.coeff[:self.n])
        if 'fingerprint' in self.features:
            fp = torch.empty(capacity, dtype=torch.int32, device=self.device)
            fp[:self.n].copy_(self.fingerprints[:self.n])
            self.fingerprints = fp
        self.recycle_pair(self.keys, self.coeff)
        self.keys, self.coeff, self.capacity = keys, coeff, capacity
        self.growths += 1

    def compact(self, cap=None, final=False):
        size = self.live if cap is None else min(self.live, cap)
        capacity = size if final else max(16, 2*size)
        keys, coeff = self.allocate_pair(capacity, final)
        capacity = keys.shape[0]
        if size:
            if size < self.live:
                ids = torch.topk(self.coeff[:self.n].abs(), size, sorted=False).indices
                keys[:size].copy_(self.keys[ids])
                coeff[:size].copy_(self.coeff[ids])
            else:
                self.counts.zero_()
                compact_variant[((self.n+127)//128,)](
                    self.keys, self.coeff, keys, coeff, self.counts, self.n, self.width, 128,
                    **self.key_args(), OUT_STRIDE=keys.stride(1), OUT_SOA='soa' in self.features and not final)
        self.recycle_pair(self.keys, self.coeff)
        self.keys, self.coeff = keys, coeff
        self.n = self.live = size
        self.capacity = capacity
        self.compactions += 1
        if not final:
            self._rebuild()

    def active_rows(self, gate):
        kind, info = self.descriptors[gate.data_ptr()]
        all_axes = bool(self.features.intersection(('select_all', 'inverted_all')))
        arity, sites = self.local_descriptors.get(gate.data_ptr(), (0, None)) if all_axes else (0, None)
        if (all_axes and not arity) or (not all_axes and not kind):
            return None, self.n
        if self.selected is None or self.selected.numel() < self.n:
            self.selected = torch.empty(max(16, self.n*5//4), dtype=torch.int32, device=self.device)
        self.counts.zero_()
        select_rows[((self.n+127)//128,)](self.keys, self.planes, self.selected,
            self.counts, info, self.n, self.plane_capacity, self.width, 128,
            KIND=kind, INVERTED=self.inverted, sites=sites, ARITY=arity)
        return self.selected, int(self.counts[0].item())

    def apply(self, gate, params, cap):
        if not self.n:
            return
        if self.n >= 2**30:
            raise ValueError('Requires fewer than 2**30 slots')
        self._reserve()
        if 2*self.n > self.table_size//2:
            self._rebuild()
        kind, info = self.descriptors[gate.data_ptr()]
        active, active_n = (None, self.n)
        if self.features.intersection(('inverted', 'select', 'inverted_all', 'select_all')) and not self.first:
            active, active_n = self.active_rows(gate)
        arity, sites = self.local_descriptors.get(gate.data_ptr(), (0, None))
        self.counts.zero_()
        if active_n:
            update_variant[((active_n+127)//128,)](
                self.keys, self.coeff, self.table, gate, params, self.counts,
                self.n, self.table_size-1, self.width, 128, **self.key_args(),
                fingerprints=self.fingerprints, FP='fingerprint' in self.features, info=info,
                LOCAL=kind if 'local' in self.features else 0,
                active=active, active_n=active_n, ACTIVE=active is not None,
                sites=sites, LOCAL_ARITY=arity if 'local_all' in self.features else 0,
                enable_fp_fusion=False)
        added, delta = self.counts.cpu().tolist()
        before = self.n
        self.n += added
        self.live += delta
        self.first = False
        if cap is not None and self.live > cap:
            self.compact(cap)
        elif self.n-self.live > max(16, self.dead_fraction*self.n):
            self.compact()
        elif added:
            self._insert_range(before)
            if self.inverted:
                self._planes(before)

    def device_gates(self, gates, cap, chunk=8):
        pos = 0
        while pos < len(gates) and self.n:
            self._reserve(max(16, (self.n*9+3)//4))
            bound = self.capacity//2
            needed_table = 1 << (max(4, 2*self.capacity)-1).bit_length()
            if needed_table > self.table_size:
                self.table_size = needed_table
                self.table = torch.full((needed_table,), -1, dtype=torch.int32, device=self.device)
                self._insert_range(0)
                self.rebuilds += 1
            state = torch.tensor([self.n, self.live, -1], dtype=torch.int32, device=self.device)
            end = min(len(gates), pos+chunk)
            for epoch in range(pos, end):
                gate, scalars = gates[epoch]
                begin_device[(1,)](state, self.counts, bound, epoch)
                update_variant[((bound+127)//128,)](self.keys, self.coeff, self.table,
                    gate, scalars, self.counts, self.n, self.table_size-1, self.width, 128,
                    state=state, epoch=epoch, DEVICE=True, active_n=bound, enable_fp_fusion=False)
                insert_device[((bound+127)//128,)](self.keys, self.table, state, self.counts,
                    bound, self.table_size-1, self.width, 128)
                advance_device[(1,)](state, self.counts, (2**31-1) if cap is None else cap, epoch)
            self.n, self.live, stop = state.cpu().tolist()
            pos = end if stop < 0 else stop
            if cap is not None and self.live > cap:
                self.compact(cap)
            elif self.n-self.live > max(16, self.dead_fraction*self.n):
                self.compact()


def evolve_options(state, operations, trunc_val=0., max_num_str=None,
                   *, mode='baseline', dead_fraction=.1, stats=None):
    from . import _gate_data, SparsePauliOp
    from ..circuit_ir import PauliRotation, SkippedOperation
    operations = tuple(operations)
    if mode == 'baseline':
        return base.evolve_step_persistent(state, operations, trunc_val, max_num_str,
                                           dead_fraction=dead_fraction, stats=stats)
    if mode not in MODES:
        raise ValueError(mode)
    if not math.isfinite(trunc_val) or trunc_val < 0:
        raise ValueError('trunc_val must be finite and nonnegative')
    if not 0 < dead_fraction < 1:
        raise ValueError('dead_fraction must be between zero and one')
    if max_num_str is not None and (not isinstance(max_num_str, int) or max_num_str < 1):
        raise ValueError('max_num_str must be a positive integer or None')
    descriptors = {}
    local_descriptors = {}
    all_axes = bool(set(mode.split('+')).intersection(('local_all', 'select_all', 'inverted_all')))
    gates = []
    for op in reversed(operations):
        if not isinstance(op, (PauliRotation, SkippedOperation)):
            raise NotImplementedError('Options support Pauli rotations only')
        if isinstance(op, PauliRotation):
            if not math.isfinite(op.theta):
                raise ValueError('theta must be finite')
            gate, scalars = _gate_data(op.pauli, float(op.theta), float(trunc_val),
                state.num_qubits, state.c_array.dtype, state.c_array.device)
            descriptors[gate.data_ptr()] = descriptor(op.pauli, state.c_array.device)
            if all_axes:
                local_descriptors[gate.data_ptr()] = local_descriptor(op.pauli, state.c_array.device)
            gates.append((gate, scalars))
    if not gates:
        return state
    with torch.cuda.device(state.c_array.device):
        storage = OptionStorage(state, dead_fraction, mode, descriptors, local_descriptors)
        if mode == 'device':
            storage.device_gates(gates, max_num_str)
        else:
            for gate, scalars in gates:
                storage.apply(gate, scalars, max_num_str)
        storage.compact(final=True)
        if stats is not None:
            stats.update(rebuilds=storage.rebuilds, compactions=storage.compactions, growths=storage.growths)
        return SparsePauliOp(storage.keys, storage.coeff, state.num_qubits)
