"""Fixed-support key-width probe; padding preserves keys, partners and gate action."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spd.triton_backend.persistent import _insert,_update
from spd.triton_backend.experiment_kernels import update_variant

from spd.triton_backend.kernels import _mix

@triton.jit
def query_hash(keys,gate,hashes,n:tl.constexpr,W:tl.constexpr):
    i=tl.program_id(0)*128+tl.arange(0,128)
    h=tl.full((128,),0x9e3779b9,tl.uint32)
    for w in tl.static_range(W):
        v=tl.load(keys+i.to(tl.int64)*W+w,i<n,0).to(tl.uint32)
        g=tl.load(gate+w).to(tl.uint32)
        h=_mix(h,v^g)
    tl.store(hashes+i,h,i<n)

@triton.jit
def lookup(keys,gate,table,hashes,out,n:tl.constexpr,W:tl.constexpr,CACHED:tl.constexpr):
    i=tl.program_id(0)*128+tl.arange(0,128)
    pending=i<n
    if CACHED:
        h=tl.load(hashes+i,pending,0).to(tl.uint32)
    else:
        h=tl.full((128,),0x9e3779b9,tl.uint32)
        for w in tl.static_range(W):
            v=tl.load(keys+i.to(tl.int64)*W+w,pending,0).to(tl.uint32)
            g=tl.load(gate+w).to(tl.uint32)
            h=_mix(h,v^g)
    slot=h & (4*n-1)
    partner=tl.full((128,),-1,tl.int32)
    while tl.sum(pending.to(tl.int32),0)>0:
        idx=tl.load(table+slot,pending,-1)
        equal=pending & (idx>=0)
        for w in tl.static_range(W):
            v=tl.load(keys+i.to(tl.int64)*W+w,pending,0).to(tl.uint32)
            g=tl.load(gate+w).to(tl.uint32)
            other=tl.load(keys+idx.to(tl.int64)*W+w,pending & (idx>=0),0).to(tl.uint32)
            equal &= other==(v^g)
        partner=tl.where(equal,idx,partner)
        pending &= (idx>=0) & ~equal
        slot=(slot+pending.to(tl.uint32)) & (4*n-1)
    tl.store(out+i,partner,i<n)


def timed(call,setup=lambda:None):
    for _ in range(3):
        setup();call()
    torch.cuda.synchronize()
    times=[]
    for _ in range(15):
        setup()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();call();b.record();b.synchronize();times.append(a.elapsed_time(b))
    return dict(median_ms=float(np.median(times)),samples_ms=times)


def main():
    torch.manual_seed(0)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=int,default=1048576)
    parser.add_argument('--words',type=int,nargs='+',default=[8,28,32,64])
    args=parser.parse_args()
    n=args.rows
    if any(w<8 or w%2 for w in args.words):
        parser.error('--words must be even and at least 8')
    if n<128 or n%128 or n & (n-1):
        parser.error('--rows must be a power of two at least 128')
    original=torch.randint(0,2**31-1,(n//2,8),dtype=torch.int32,device='cuda').repeat_interleave(2,0)
    original[:,0]=torch.arange(n,device='cuda',dtype=torch.int32)//2
    original[1::2,0]|=-2147483648
    original[:,4]|=-2147483648
    coeff=torch.ones(n,dtype=torch.float64,device='cuda')
    counts=torch.zeros(2,dtype=torch.int32,device='cuda')
    scalars=torch.tensor([math.cos(.02),math.sin(.02),0.],dtype=torch.float64,device='cuda')
    sites=torch.tensor([0,31,1,0],dtype=torch.int32,device='cuda')
    rows=[]
    for w in args.words:
        keys=torch.zeros((n,w),dtype=torch.int32,device='cuda')
        keys[:,:4]=original[:,:4];keys[:,w//2:w//2+4]=original[:,4:]
        gate=torch.zeros(w,dtype=torch.int32,device='cuda');gate[0]=-2147483648
        table=torch.empty(4*n,dtype=torch.int32,device='cuda')
        def insert(): _insert[((n+127)//128,)](keys,table,0,n,4*n-1,w,128)
        insertion=timed(insert,lambda:table.fill_(-1))
        table.fill_(-1);insert()
        assert int((table>=0).sum())==n
        def setup(): coeff.fill_(1);counts.zero_()
        def generic(): _update[((n+127)//128,)](keys,coeff,table,gate,scalars,counts,n,4*n-1,w,128,enable_fp_fusion=False)
        def local(): update_variant[((n+127)//128,)](keys,coeff,table,gate,scalars,counts,n,4*n-1,w,128,sites=sites,LOCAL_ARITY=1,enable_fp_fusion=False)
        generic_time=timed(generic,setup)
        expected=coeff.clone()
        assert counts.tolist()==[0,0]
        local_time=timed(local,setup)
        assert counts.tolist()==[0,0]
        assert torch.equal(coeff,expected)
        assert abs(float(coeff.square().sum())/n-1)<1e-12
        hashes=torch.empty(n,dtype=torch.int32,device='cuda')
        out=torch.empty_like(hashes)
        hash_time=timed(lambda:query_hash[(n//128,)](keys,gate,hashes,n,w))
        lookup_time=timed(lambda:lookup[(n//128,)](keys,gate,table,hashes,out,n,w,False))
        assert torch.equal(out,torch.arange(n,device='cuda',dtype=torch.int32)^1)
        cached_time=timed(lambda:lookup[(n//128,)](keys,gate,table,hashes,out,n,w,True))
        assert torch.equal(out,torch.arange(n,device='cuda',dtype=torch.int32)^1)
        rows.append(dict(words=w,insert=insertion,update=generic_time,local_update=local_time,
            query_hash=hash_time,lookup=lookup_time,ideal_cached_query_lookup=cached_time))
        print(w,{k:v['median_ms'] for k,v in rows[-1].items() if isinstance(v,dict)},flush=True)
    out=ROOT/f'benchmarks/results/persistent_key_scaling_20260911/key_width_{n}_{'_'.join(map(str,args.words))}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(rows=n,anti_fraction=1.,partner_hit_fraction=1.,gate='X0',padding_only=True,
        gpu=torch.cuda.get_device_name(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['spd/triton_backend/persistent.py','spd/triton_backend/experiment_kernels.py','benchmarks/benchmark_persistent_key_width.py']},results=rows),indent=2)+'\n')

if __name__=='__main__':main()
