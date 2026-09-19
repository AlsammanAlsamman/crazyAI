"""kmer.* - compile a C k-mer-counting kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int n, const char *seq, int k, uint64_t *counts);
    // counts every length-k substring of seq (over A/C/G/T) into counts[code],
    // where code is the k-mer's bases packed 2 bits each, high base first.
    // counts has 4^k entries and arrives pre-zeroed.

The harness (embedded below) fixes k=8 (4^8 = 65536 counts, trivial memory),
generates a random ACGT sequence, checks the candidate's count table against
a naive re-encode-every-window reference (exact integer match, no tolerance -
counting is deterministic), and times both, the same shape as
kernel.py's/alignment.py's harnesses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("void kernel(int n, const char *seq, int k, uint64_t *counts);  /* counts every length-k substring of "
           "seq (over A/C/G/T) into counts[code] - code packs the k bases 2 bits each, high base first; counts has "
           "4^k entries and arrives pre-zeroed */")

EXAMPLE = """#include <stdint.h>
static int base_code(char c) {
    switch (c) { case 'A': return 0; case 'C': return 1; case 'G': return 2; default: return 3; }
}
void kernel(int n, const char *seq, int k, uint64_t *counts) {
    for (int i = 0; i + k <= n; i++) {
        uint64_t code = 0;
        for (int j = 0; j < k; j++) code = (code << 2) | (uint64_t)base_code(seq[i + j]);
        counts[code]++;
    }
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
void kernel(int n, const char *seq, int k, uint64_t *counts);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int base_code(char c){switch(c){case 'A':return 0;case 'C':return 1;case 'G':return 2;default:return 3;}}
static void ref_count(int n,const char*seq,int k,uint64_t*counts){
 for(int i=0;i+k<=n;i++){uint64_t code=0;for(int j=0;j<k;j++)code=(code<<2)|(uint64_t)base_code(seq[i+j]);counts[code]++;}}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
typedef void(*kfn)(int,const char*,int,uint64_t*);
static double timeit(kfn f,int n,const char*seq,int k,uint64_t*counts,int csize,double budget){
 memset(counts,0,(size_t)csize*sizeof(uint64_t));f(n,seq,k,counts);
 double best=1e30,tot=0;int reps=0;
 while(reps<3||(tot<budget&&reps<40)){
  memset(counts,0,(size_t)csize*sizeof(uint64_t));
  double t0=now();f(n,seq,k,counts);double dt=now()-t0;if(dt<best)best=dt;tot+=dt;reps++;}
 return best;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 int K=8;int csize=1;for(int i=0;i<K;i++)csize*=4;
 const char bases[4]={'A','C','G','T'};
 for(int a=2;a<argc;a++){int n=atoi(argv[a]);
  char*seq=malloc((size_t)n+1);uint64_t s=999+(uint64_t)n;
  for(int i=0;i<n;i++)seq[i]=bases[xs(&s)&3];seq[n]=0;
  uint64_t*ref=malloc((size_t)csize*sizeof(uint64_t));
  uint64_t*out=malloc((size_t)csize*sizeof(uint64_t));
  double tref=timeit(ref_count,n,seq,K,ref,csize,budget/2);
  double tk=timeit(kernel,n,seq,K,out,csize,budget);
  int mismatches=0;uint64_t maxdiff=0;
  for(int i=0;i<csize;i++){uint64_t a1=ref[i],b1=out[i];uint64_t d=a1>b1?a1-b1:b1-a1;if(d){mismatches++;if(d>maxdiff)maxdiff=d;}}
  printf("{\"n\":%d,\"k\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"mismatches\":%d,\"max_diff\":%llu}\n",
        n,K,tk,tref,mismatches,(unsigned long long)maxdiff);
  fflush(stdout);free(seq);free(ref);free(out);}
 return 0;}
"""


@tool("kmer", "measure")
def contract() -> dict:
    """The fixed C signature every k-mer-counting idea must implement, an example (naive) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp",
            "notes": "k is fixed at 8 by the harness (counts has 4^8 = 65536 entries, pre-zeroed by the caller "
                     "before every timed call). seq is a random string over A/C/G/T."}


@tool("kmer", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C k-mer-counting kernel, check its count table against a naive reference (exact match) and time it.
    Returns error, speedup vs. the naive reference, and value = speedup x (1.0 if the table matches exactly else 0.0).

    Args:
        source: Complete C source defining `void kernel(int n, const char *seq, int k, uint64_t *counts)`.
        sizes: Sequence lengths to run (default [4096, 65536, 262144]).
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the kmer tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [4096, 65536, 262144])]
    if any(s < 16 or s > 2_000_000 for s in sizes):
        return {"error": "sizes must be between 16 and 2,000,000"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_kmer_"))
    (work / "kernel.c").write_text(source, encoding="utf-8")
    (work / "harness.c").write_text(_HARNESS, encoding="utf-8")
    exe = work / "bench"
    cc = subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(exe), str(work / "kernel.c"), str(work / "harness.c")],
                        capture_output=True, text=True, timeout=120)
    if cc.returncode != 0:
        return {"error": "compile failed", "compiler_output": cc.stderr[-3000:], "status": "COMPILE_ERROR"}
    try:
        run = subprocess.run([str(exe), str(budget)] + [str(s) for s in sizes], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"error": "kernel timed out (300 s)", "status": "TIMEOUT"}
    if run.returncode != 0:
        return {"error": f"kernel crashed (exit {run.returncode})", "stderr": run.stderr[-2000:], "status": "CRASH"}
    rows = [json.loads(l) for l in run.stdout.splitlines() if l.startswith("{")]
    for r in rows:
        r["correct"] = r["mismatches"] == 0
        r["speedup_vs_naive"] = round(r["ref_time"] / r["time"], 3) if r["ref_time"] > 0 else None
        r["status"] = "exact" if r["correct"] else "WRONG"
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round((big["speedup_vs_naive"] or 0.0) * (1.0 if big["correct"] else 0.0), 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "status": big["status"],
            "speedup_vs_naive": big["speedup_vs_naive"], "value": value, "prediction_target": big["speedup_vs_naive"],
            "reading": "value = speedup vs. the naive re-encode-every-window O(n*k) reference, 0 if the count table doesn't match exactly."}
