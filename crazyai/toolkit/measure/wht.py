"""wht.* - compile a C Walsh-Hadamard-transform kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int n, const double *in, double *out);
    // Walsh-Hadamard transform of a length-n (power of two) real sequence:
    // out[k] = sum_j in[j] * (-1)^popcount(j & k)

The harness (embedded below) checks the kernel against a naive O(n^2)
direct-matrix reference and times it, the same shape as fft.py's harness
(3-tier exactness on relative error, value = speedup x exactness) - but
real-valued, no complex numbers, no trig: only additions and subtractions,
so no volatile sink is needed (heap-buffer writes are never eliminated by
GCC, the same finding fft.py already relies on).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("void kernel(int n, const double *in, double *out);  /* Walsh-Hadamard transform of a length-n "
           "(power of two) real sequence: out[k] = sum_j in[j] * (-1)^popcount(j & k); out may be uninitialised "
           "on entry */")

EXAMPLE = """static int popcount(int x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }
void kernel(int n, const double *in, double *out) {
    for (int k = 0; k < n; k++) {
        double s = 0;
        for (int j = 0; j < n; j++) s += in[j] * ((popcount(j & k) & 1) ? -1.0 : 1.0);
        out[k] = s;
    }
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
void kernel(int n, const double *in, double *out);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int popcount(int x){int c=0;while(x){c+=x&1;x>>=1;}return c;}
static void naive_wht(int n,const double*in,double*out){
 for(int k=0;k<n;k++){double s=0;
  for(int j=0;j<n;j++)s+=in[j]*((popcount(j&k)&1)?-1.0:1.0);
  out[k]=s;}}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
typedef void(*whtfn)(int,const double*,double*);
static double timeit(whtfn f,int n,const double*in,double*out,double budget){
 f(n,in,out);double best=1e30,tot=0;int reps=0;
 while(reps<3||(tot<budget&&reps<40)){double t0=now();f(n,in,out);double dt=now()-t0;if(dt<best)best=dt;tot+=dt;reps++;}
 return best;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 for(int a=2;a<argc;a++){int n=atoi(argv[a]);
  double*in=malloc((size_t)n*8),*ref=malloc((size_t)n*8),*out=malloc((size_t)n*8);
  uint64_t s=4321+(uint64_t)n;
  for(int i=0;i<n;i++)in[i]=(double)(xs(&s)>>11)/9007199254740992.0-0.5;
  naive_wht(n,in,ref);
  memset(out,0x7f,(size_t)n*8);
  kernel(n,in,out);
  double num=0,den=0;int finite=1;
  for(int i=0;i<n;i++){double d=out[i]-ref[i];
   if(!isfinite(d)){finite=0;break;}
   num+=d*d;den+=ref[i]*ref[i];}
  double rel=finite?sqrt(num/(den>0?den:1)):INFINITY;
  double tk=timeit(kernel,n,in,out,budget);
  double tref=n<=1024?timeit(naive_wht,n,in,ref,budget/2):-1;
  printf("{\"n\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"rel_err\":%.3e}\n",n,tk,tref,rel);
  fflush(stdout);free(in);free(ref);free(out);}
 return 0;}
"""


def _exactness(rel: float) -> float:
    if rel < 1e-9:
        return 1.0
    if rel < 1e-3:
        return 0.5
    if rel < 1e-1:
        return 0.1
    return 0.0


@tool("wht", "measure")
def contract() -> dict:
    """The fixed C signature every Walsh-Hadamard idea must implement, an example (naive) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": "n is a power of two between 16 and 4096 in the harness. out is scratch you fill in; in must not be modified."}


@tool("wht", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C Walsh-Hadamard-transform kernel, check it against a naive O(n^2) direct-matrix reference and
    time it. Returns error, speedup vs. the reference, and value = speedup_vs_naive x exactness.

    Args:
        source: Complete C source defining `void kernel(int n, const double *in, double *out)`.
        sizes: Sequence lengths to run, each a power of two (default [64, 256, 1024]).
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the wht tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [64, 256, 1024])]
    if any(s < 4 or s > 8192 or (s & (s - 1)) for s in sizes):
        return {"error": "sizes must be powers of two between 4 and 8192"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_wht_"))
    (work / "kernel.c").write_text(source, encoding="utf-8")
    (work / "harness.c").write_text(_HARNESS, encoding="utf-8")
    exe = work / "bench"
    cc = subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(exe), str(work / "kernel.c"), str(work / "harness.c"), "-lm"],
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
        r["speedup_vs_naive"] = round(r["ref_time"] / r["time"], 3) if r["ref_time"] > 0 else None
        r["exactness"] = _exactness(r["rel_err"])
        r["status"] = "exact" if r["exactness"] == 1.0 else "approx" if r["exactness"] > 0 else "WRONG"
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round((big["speedup_vs_naive"] or 0.0) * big["exactness"], 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "rel_err": big["rel_err"], "status": big["status"],
            "speedup_vs_naive": big["speedup_vs_naive"], "value": value, "prediction_target": big["speedup_vs_naive"],
            "reading": "value = speedup vs. the naive O(n^2) direct-matrix WHT x exactness (1.0 relative error < 1e-9)."}
