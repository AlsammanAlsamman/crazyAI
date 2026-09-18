"""alignment.* - compile a C global-sequence-alignment kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    int kernel(int n, const char *a, const char *b);   // global alignment score, a/b length-n strings over ACGT

The harness (embedded below) checks the kernel's score against a reference
Needleman-Wunsch DP (match=+1, mismatch=-1, gap=-2) and times it. Unlike the
matmul kernel tool, correctness here is binary - an alignment score is a
single integer, either it matches the reference exactly or the kernel is wrong.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("int kernel(int n, const char *a, const char *b);  /* global alignment score: a, b are length-n "
           "strings over {A,C,G,T}; match=+1, mismatch=-1, gap=-2, standard Needleman-Wunsch scoring */")

EXAMPLE = """#include <stdlib.h>
#define MATCH 1
#define MISMATCH -1
#define GAP -2
int kernel(int n, const char *a, const char *b) {
    int *dp = malloc((size_t)(n + 1) * (n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) dp[i * (n + 1) + 0] = i * GAP;
    for (int j = 0; j <= n; j++) dp[0 * (n + 1) + j] = j * GAP;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) {
            int diag = dp[(i - 1) * (n + 1) + (j - 1)] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int up = dp[(i - 1) * (n + 1) + j] + GAP;
            int left = dp[i * (n + 1) + (j - 1)] + GAP;
            int best = diag;
            if (up > best) best = up;
            if (left > best) best = left;
            dp[i * (n + 1) + j] = best;
        }
    int result = dp[n * (n + 1) + n];
    free(dp);
    return result;
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#define MATCH 1
#define MISMATCH -1
#define GAP -2
int kernel(int n, const char *a, const char *b);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int nw_ref(int n,const char*a,const char*b,int*dp){
 for(int i=0;i<=n;i++)dp[i*(n+1)+0]=i*GAP;for(int j=0;j<=n;j++)dp[0*(n+1)+j]=j*GAP;
 for(int i=1;i<=n;i++)for(int j=1;j<=n;j++){
  int diag=dp[(i-1)*(n+1)+(j-1)]+(a[i-1]==b[j-1]?MATCH:MISMATCH);
  int up=dp[(i-1)*(n+1)+j]+GAP,left=dp[i*(n+1)+(j-1)]+GAP;
  int best=diag;if(up>best)best=up;if(left>best)best=left;dp[i*(n+1)+j]=best;}
 return dp[n*(n+1)+n];}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 for(int ai=2;ai<argc;ai++){int n=atoi(argv[ai]);
  char*a=malloc(n+1),*b=malloc(n+1);uint64_t s=1234+n;const char letters[4]={'A','C','G','T'};
  for(int i=0;i<n;i++)a[i]=letters[(xs(&s)>>13)&3];for(int i=0;i<n;i++)b[i]=letters[(xs(&s)>>13)&3];a[n]=0;b[n]=0;
  int*dp=malloc((size_t)(n+1)*(n+1)*sizeof(int));
  int ref_score=nw_ref(n,a,b,dp);
  double best_ref=1e30,tot=0;int reps=0;
  while(reps<3||(tot<budget/2&&reps<40)){double t0=now();nw_ref(n,a,b,dp);double dt=now()-t0;if(dt<best_ref)best_ref=dt;tot+=dt;reps++;}
  int cand_score=kernel(n,a,b);
  double best_k=1e30;tot=0;reps=0;
  while(reps<3||(tot<budget&&reps<40)){double t0=now();kernel(n,a,b);double dt=now()-t0;if(dt<best_k)best_k=dt;tot+=dt;reps++;}
  printf("{\"n\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"ref_score\":%d,\"cand_score\":%d,\"correct\":%s}\n",
   n,best_k,best_ref,ref_score,cand_score,cand_score==ref_score?"true":"false");
  fflush(stdout);free(a);free(b);free(dp);}
 return 0;}
"""


@tool("alignment", "measure")
def contract() -> dict:
    """The fixed C signature every sequence-alignment idea must implement, an example kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": ("a/b are length-n strings over {A,C,G,T}. n is a sequence length, not a matrix side. Score must "
                     "exactly match the Needleman-Wunsch reference (match=+1, mismatch=-1, gap=-2) - alignment score is "
                     "a single integer, no partial credit for a near-miss. You may allocate scratch memory, use OpenMP "
                     "or SIMD.")}


@tool("alignment", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C global-alignment kernel, check its score against a Needleman-Wunsch reference and time it.
    Returns whether it's exactly correct, speedup vs. the reference DP, and value = speedup_vs_dp if correct else 0.

    Args:
        source: Complete C source defining `int kernel(int n, const char *a, const char *b)`.
        sizes: Sequence lengths to run (default [64, 256, 1024]).
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the alignment tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [64, 256, 1024])]
    if any(s < 4 or s > 4096 for s in sizes):
        return {"error": "sizes must be between 4 and 4096"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_alignment_"))
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
        r["speedup_vs_dp"] = round(r["ref_time"] / r["time"], 3) if r["time"] > 0 else None
        r["status"] = "exact" if r["correct"] else "WRONG"
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round(big["speedup_vs_dp"], 4) if big["correct"] and big["speedup_vs_dp"] else 0.0
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "status": big["status"], "speedup_vs_dp": big["speedup_vs_dp"],
            "value": value, "prediction_target": big["speedup_vs_dp"],
            "reading": "value = speedup vs. the Needleman-Wunsch reference, or 0 if the score is wrong - correctness is binary here, not a tolerance."}
