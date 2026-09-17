"""kernel.* - compile a C matrix-multiplication kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int n, const double *A, const double *B, double *C);   // C = A * B, row-major n x n

The harness (embedded below) checks the kernel against a naive reference,
times it next to the naive loop and a cache-blocked baseline, and reports
GFLOP/s, error and a value = speedup-over-baseline x exactness.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = "void kernel(int n, const double *A, const double *B, double *C);  /* C = A*B, row-major, n x n, C may be overwritten */"

EXAMPLE = """#include <string.h>
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
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
void kernel(int n, const double *A, const double *B, double *C);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void naive(int n,const double*A,const double*B,double*C){for(int i=0;i<n;i++)for(int j=0;j<n;j++){double s=0;for(int k=0;k<n;k++)s+=A[(size_t)i*n+k]*B[(size_t)k*n+j];C[(size_t)i*n+j]=s;}}
static void blocked(int n,const double*A,const double*B,double*C){memset(C,0,(size_t)n*n*sizeof(double));const int BS=64;
 for(int ii=0;ii<n;ii+=BS)for(int kk=0;kk<n;kk+=BS)for(int jj=0;jj<n;jj+=BS){int ie=ii+BS<n?ii+BS:n,je=jj+BS<n?jj+BS:n,ke=kk+BS<n?kk+BS:n;
 for(int i=ii;i<ie;i++)for(int k=kk;k<ke;k++){double a=A[(size_t)i*n+k];for(int j=jj;j<je;j++)C[(size_t)i*n+j]+=a*B[(size_t)k*n+j];}}}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
static double timeit(void(*f)(int,const double*,const double*,double*),int n,const double*A,const double*B,double*C,double budget){
 f(n,A,B,C);double best=1e30,tot=0;int reps=0;while(reps<3||(tot<budget&&reps<40)){double t0=now();f(n,A,B,C);double dt=now()-t0;if(dt<best)best=dt;tot+=dt;reps++;}return best;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 for(int a=2;a<argc;a++){int n=atoi(argv[a]);size_t N=(size_t)n*n;
  double*A=malloc(N*8),*B=malloc(N*8),*C=malloc(N*8),*R=malloc(N*8);uint64_t s=1234+n;
  for(size_t i=0;i<N;i++)A[i]=(double)(xs(&s)>>11)/9007199254740992.0;for(size_t i=0;i<N;i++)B[i]=(double)(xs(&s)>>11)/9007199254740992.0;
  naive(n,A,B,R);memset(C,0x7f,N*8);kernel(n,A,B,C);
  double ma=0,num=0,den=0;int finite=1;for(size_t i=0;i<N;i++){double d=C[i]-R[i];if(!isfinite(d)){finite=0;break;}if(fabs(d)>ma)ma=fabs(d);num+=d*d;den+=R[i]*R[i];}
  double rel=finite?sqrt(num/(den>0?den:1)):INFINITY;
  double tk=timeit(kernel,n,A,B,C,budget),tn=n<=512?timeit(naive,n,A,B,C,budget/2):-1,tb=timeit(blocked,n,A,B,C,budget/2);
  printf("{\"n\":%d,\"time\":%.9g,\"gflops\":%.4f,\"naive_time\":%.9g,\"blocked_time\":%.9g,\"max_abs_err\":%.3e,\"rel_fro_err\":%.3e}\n",
   n,tk,2.0*n*N/tk*1e-9,tn,tb,ma,rel);fflush(stdout);free(A);free(B);free(C);free(R);}
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


@tool("kernel", "measure")
def contract() -> dict:
    """The fixed C signature every matrix-multiplication idea must implement, an example kernel, and the compiler flags used.

    """
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": "Row-major. n is a power of two between 16 and 1024 in the harness. You may allocate scratch memory, use OpenMP, AVX intrinsics (immintrin.h) or any exact or approximate scheme; the harness reports the error either way."}


@tool("kernel", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C kernel with the fixed contract, check it against a naive reference and time it next to the naive loop and a 64x64 cache-blocked baseline. Returns GFLOP/s, error, speedups and value = speedup_over_blocked x exactness.

    Args:
        source: Complete C source defining `void kernel(int n, const double *A, const double *B, double *C)`.
        sizes: Matrix sizes to run (default [64, 256, 512]).
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the kernel tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [64, 256, 512])]
    if any(s < 4 or s > 2048 for s in sizes):
        return {"error": "sizes must be between 4 and 2048"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_kernel_"))
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
        r["speedup_vs_naive"] = round(r["naive_time"] / r["time"], 3) if r["naive_time"] > 0 else None
        r["speedup_vs_blocked"] = round(r["blocked_time"] / r["time"], 3)
        r["exactness"] = _exactness(r["rel_fro_err"])
        r["status"] = "exact" if r["exactness"] == 1.0 else "approx" if r["exactness"] > 0 else "WRONG"
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round(big["speedup_vs_blocked"] * big["exactness"], 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "gflops": big["gflops"], "rel_fro_err": big["rel_fro_err"],
            "status": big["status"], "speedup_vs_naive": big["speedup_vs_naive"], "speedup_vs_blocked": big["speedup_vs_blocked"],
            "value": value, "reading": ("value > 1 means faster than a cache-blocked loop and exact; "
                                        "> 3 beats a vectorised kernel; ~10 is OpenBLAS territory on one core")}
