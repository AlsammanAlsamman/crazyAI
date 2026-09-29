"""Stronger quality check for hash kernels than the harness's 200-trial avalanche score.

The `hash.bench` harness calls a kernel "exact" when its mean avalanche over 200 random single-bit
flips is > 0.9 - a weak test that a fast-but-poorly-mixing hash could pass. This runs the kind of
checks SMHasher uses:

- avalanche matrix: for many random keys, flip every input bit (every 32nd on 1024-byte keys) and record, for every
  (input bit, output bit) pair, how often the output bit changes. worst_bias = max |p - 0.5|
  over the whole matrix. Sampling noise alone gives a worst bias of
  roughly 0.03-0.07 here, so read it against the two reference kernels in examples/hash_refs/
  (FNV-1a, known weak; a murmur3-finalizer hash, known strong).
- sparse/structured keys: 2^20 sequential little-endian counters in 16-byte buffers, and all 8128
  16-byte keys with exactly two bits set; count full 64-bit and low-32-bit collisions vs. the birthday
  expectation.

    python examples/11_hash_quality.py <kernel.c> [<kernel.c> ...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
uint64_t kernel(const unsigned char *data, size_t len);
static uint64_t s = 0x9E3779B97F4A7C15ULL;
static uint64_t rnd(void){ s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static int cmp(const void*a,const void*b){ uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return x<y?-1:x>y; }
static long collisions(uint64_t *h, long n, uint64_t mask){
  for(long i=0;i<n;i++) h[i]&=mask; qsort(h,n,8,cmp); long c=0; for(long i=1;i<n;i++) c+=h[i]==h[i-1]; return c; }
int main(void){
  int lens[3]={16,64,1024}; int keys[3]={4000,1500,1000}; int steps[3]={1,1,32};
  printf("{\"avalanche\":[");
  for(int li=0;li<3;li++){
    int L=lens[li], K=keys[li], ST=steps[li], IB=L*8; static unsigned char buf[1024];
    long *cnt=calloc((size_t)IB*64,sizeof(long));
    for(int k=0;k<K;k++){
      for(int i=0;i<L;i++) buf[i]=(unsigned char)rnd();
      uint64_t h0=kernel(buf,L);
      for(int b=0;b<IB;b+=ST){ buf[b>>3]^=1u<<(b&7); uint64_t d=h0^kernel(buf,L); buf[b>>3]^=1u<<(b&7);
        for(int o=0;o<64;o++) cnt[(size_t)b*64+o]+=(d>>o)&1; } }
    double worst=0, sum=0; long cells=0; for(size_t i=0;i<(size_t)IB*64;i++){ if(((i/64)%ST)!=0) continue; cells++; double p=(double)cnt[i]/K, e=p>0.5?p-0.5:0.5-p; sum+=e; if(e>worst) worst=e; }
    printf("%s{\"len\":%d,\"keys\":%d,\"worst_bias\":%.4f,\"mean_bias\":%.4f}",li?",":"",L,K,worst,sum/cells);
    free(cnt); }
  long N=1L<<20; uint64_t *h=malloc(N*8); unsigned char k16[16];
  for(uint64_t i=0;i<(uint64_t)N;i++){ memset(k16,0,16); memcpy(k16,&i,8); h[i]=kernel(k16,16); }
  long c64=collisions(h,N,~0ULL);
  for(uint64_t i=0;i<(uint64_t)N;i++){ memset(k16,0,16); memcpy(k16,&i,8); h[i]=kernel(k16,16); }
  long c32=collisions(h,N,0xFFFFFFFFULL);
  long m=0; for(int a=0;a<128&&m<N;a++) for(int b=a+1;b<128&&m<N;b++){ memset(k16,0,16); k16[a>>3]|=1u<<(a&7); k16[b>>3]|=1u<<(b&7); h[m++]=kernel(k16,16); }
  long t64=collisions(h,m,~0ULL);
  printf("],\"sequential_keys\":%ld,\"seq_coll64\":%ld,\"seq_coll32\":%ld,\"seq_coll32_expected\":%.1f,\"twobit_keys\":%ld,\"twobit_coll64\":%ld}\n",
         N,c64,c32,(double)N*(N-1)/2/4294967296.0,m,t64);
  return 0; }
"""


def check(kernel: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix="hashq_"))
    (work / "h.c").write_text(HARNESS, encoding="utf-8")
    exe = work / "hq.exe"
    cc = subprocess.run(["gcc", "-O2", "-march=native", "-fopenmp", "-o", str(exe), str(kernel), str(work / "h.c")],
                        capture_output=True, text=True)
    if cc.returncode:
        return {"kernel": str(kernel), "error": cc.stderr[-800:]}
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=1800)
    out = json.loads(run.stdout) if run.returncode == 0 else {"error": run.stderr[-800:]}
    out["kernel"] = str(kernel)
    if "avalanche" in out:
        out["worst_bias"] = max(a["worst_bias"] for a in out["avalanche"])
        out["pass"] = out["worst_bias"] < 0.08 and out["seq_coll64"] == 0 and out["twobit_coll64"] == 0
    return out


if __name__ == "__main__":
    for k in sys.argv[1:]:
        print(json.dumps(check(Path(k))), flush=True)
