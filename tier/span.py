#!/usr/bin/env python3
"""Predict Metal buffer spans for a placement BEFORE running, so we never OOM again.

llama.cpp computes the Metal buffer span PER SHARD as (max offset+size - min offset)
over the tensors assigned to that device in that shard. Sparse placement inside a big
shard therefore costs span, not bytes. Limits on this machine:
  per-span    (none: ggml splits into views)
  total       recommendedMaxWorkingSetSize = 120.0 GiB
usage: span.py "<ot-spec>" <ngl> [model-path-or-glob]
   env SPAN_GLOB also accepted; defaults to the Q8 5-shard split.
"""
import re, sys, struct, glob, os, collections
BLK={2:(32,18),3:(32,20),8:(32,34),14:(256,210)}
def u32(f): return struct.unpack('<I',f.read(4))[0]
def u64(f): return struct.unpack('<Q',f.read(8))[0]
def rstr(f): return f.read(u64(f)).decode('utf-8','replace')
def skipval(f,t):
    if t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t in (4,5,6): f.read(4)
    elif t in (10,11,12): f.read(8)
    elif t==8: rstr(f)
    elif t==9:
        et=u32(f); n=u64(f)
        for _ in range(n): skipval(f,et)
def nbytes(dims,typ):
    ne=1
    for d in dims: ne*=d
    if typ in BLK:
        be,bb=BLK[typ]; return ne//be*bb
    return ne*(2 if typ in (1,30) else 4)
def load():
    out=[]
    pat=(sys.argv[3] if len(sys.argv)>3 else
         os.environ.get('SPAN_GLOB','./q8-*-of-00005.gguf'))
    for p in sorted(glob.glob(pat)):
        with open(p,'rb') as f:
            f.read(4); u32(f); nt=u64(f); nkv=u64(f)
            for _ in range(nkv): rstr(f); skipval(f,u32(f))
            base=[]
            for _ in range(nt):
                nm=rstr(f); nd=u32(f); dims=[u64(f) for _ in range(nd)]
                ty=u32(f); off=u64(f); base.append((nm,dims,ty,off))
            for nm,dims,ty,off in base:
                out.append((os.path.basename(p),nm,off,nbytes(dims,ty)))
    return out
def place(name, rules, ngl, n_layer=48):
    for pat,dev in rules:                       # first match wins (loader breaks)
        if re.search(pat,name): return dev
    m=re.match(r'blk\.(\d+)\.',name)
    if m:                                       # -ngl puts the LAST ngl layers on GPU
        return 'MTL0' if int(m.group(1)) >= n_layer-ngl else 'CPU'
    return 'MTL0' if ngl>0 else 'CPU'           # non-layer tensors follow output device
def main():
    spec,ngl=sys.argv[1],int(sys.argv[2])
    rules=[]
    for part in spec.split(','):
        if not part.strip(): continue
        pat,dev=part.rsplit('=',1); rules.append((pat,dev))
    rows=load(); GiB=1024**3
    per=collections.defaultdict(lambda:[None,None,0])
    for sh,nm,off,nb in rows:
        if place(nm,rules,ngl)!='MTL0': continue
        d=per[sh]
        d[0]=off if d[0] is None else min(d[0],off)
        d[1]=off+nb if d[1] is None else max(d[1],off+nb)
        d[2]+=nb
    tot_span=tot_bytes=0; worst=0
    print(f"{'shard':<26}{'span GiB':>10}{'bytes GiB':>11}{'waste':>9}")
    for sh in sorted(per):
        lo,hi,b=per[sh]; span=hi-lo
        tot_span+=span; tot_bytes+=b; worst=max(worst,span)
        print(f"{sh:<26}{span/GiB:10.2f}{b/GiB:11.2f}{(span-b)/GiB:9.2f}")
    print(f"{'TOTAL':<26}{tot_span/GiB:10.2f}{tot_bytes/GiB:11.2f}")
    ok=True
    # NOTE: no per-span limit. ggml splits spans > maxBufferLength into
    # overlapping views (ggml-metal-device.m:1753-1780), so 80.64 GiB is NOT
    # a failure point. Only TOTAL allocation matters (physical wired ceiling).
    if tot_span/GiB > 120.0: print(f"FAIL: total span {tot_span/GiB:.2f} GiB > working set 120.0"); ok=False
    if tot_span/GiB > 100.0 and ok: print("WARN: total span >100 GiB, close to the limit")
    print("VERDICT: SAFE" if ok else "VERDICT: UNSAFE - DO NOT RUN")
    sys.exit(0 if ok else 1)
main()
