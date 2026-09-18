"""nim.* - compile a C Nim move-chooser, check it against Bouton's theorem, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int n, const int *piles, int *out_pile, int *out_remove);
    // choose a move: reduce piles[out_pile] by out_remove (1..piles[out_pile])

The harness (embedded below) generates random Nim positions, checks whether the
kernel's move is legal and - when the position is winnable (XOR of all piles is
nonzero) - whether it actually wins (drives the XOR to zero, per Bouton's 1901
theorem), and times the decision. Correctness here is "fraction of winnable
positions where a winning move was found," not a single pass/fail.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("void kernel(int n, const int *piles, int *out_pile, int *out_remove);  /* choose a move: reduce "
           "piles[*out_pile] by *out_remove (1 <= *out_remove <= piles[*out_pile]); n piles, each piles[i] >= 1 */")

EXAMPLE = """void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    int xor_all = 0;
    for (int i = 0; i < n; i++) xor_all ^= piles[i];
    if (xor_all == 0) { *out_pile = 0; *out_remove = 1; return; }  /* no winning move exists; take anything legal */
    for (int i = 0; i < n; i++) {
        int target = piles[i] ^ xor_all;
        if (target < piles[i]) { *out_pile = i; *out_remove = piles[i] - target; return; }
    }
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
void kernel(int n, const int *piles, int *out_pile, int *out_remove);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
static void ref_move(int n,const int*piles,int xor_all,int*out_pile,int*out_remove){
 for(int i=0;i<n;i++){int target=piles[i]^xor_all;if(target<piles[i]){*out_pile=i;*out_remove=piles[i]-target;return;}}
 *out_pile=0;*out_remove=1;}
#define TRIALS 300
#define MAXPILE 31
#define BATCH 200000
/* a single move decision is a few ns - too fast for per-call clock_gettime resolution,
   so timing runs BATCH calls back-to-back on one fixed position and divides by BATCH. */
static double time_batch_kernel(int n,const int*piles){
 int op,orm;volatile int sink=0;double t0=now();
 for(int b=0;b<BATCH;b++){kernel(n,piles,&op,&orm);sink+=op+orm;}
 return (now()-t0)/BATCH;}
static double time_batch_ref(int n,const int*piles,int xor_all){
 int rp,rr;volatile int sink=0;double t0=now();
 for(int b=0;b<BATCH;b++){ref_move(n,piles,xor_all,&rp,&rr);sink+=rp+rr;}
 return (now()-t0)/BATCH;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;(void)budget;
 for(int ai=2;ai<argc;ai++){int n=atoi(argv[ai]);uint64_t s=1234+n;
  int piles[256];int correct=0,legal=0,n_positions=0;
  for(int trial=0;trial<TRIALS;trial++){
   int xor_all=0;for(int i=0;i<n;i++){piles[i]=1+(int)(xs(&s)%MAXPILE);xor_all^=piles[i];}
   int op=-1,orm=-1;kernel(n,piles,&op,&orm);
   if(xor_all!=0){n_positions++;
    int ok=(op>=0&&op<n&&orm>=1&&orm<=piles[op]);
    if(ok){legal++;int newv=piles[op]-orm;int newx=xor_all^piles[op]^newv;if(newx==0)correct++;}}}
  int tpiles[256];uint64_t ts=99+n;int txor=0;for(int i=0;i<n;i++){tpiles[i]=1+(int)(xs(&ts)%MAXPILE);txor^=tpiles[i];}
  double best_t=time_batch_kernel(n,tpiles),best_ref=time_batch_ref(n,tpiles,txor);
  double frac_correct=n_positions>0?(double)correct/n_positions:1.0;
  double frac_legal=n_positions>0?(double)legal/n_positions:1.0;
  printf("{\"n\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"trials\":%d,\"n_positions\":%d,\"frac_correct\":%.4f,\"frac_legal\":%.4f}\n",
   n,best_t,best_ref,TRIALS,n_positions,frac_correct,frac_legal);fflush(stdout);}
 return 0;}
"""


@tool("nim", "measure")
def contract() -> dict:
    """The fixed C signature every Nim move-choosing idea must implement, an example (optimal) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": ("n is the number of piles (a game size, not an array length in bytes). Each piles[i] is a "
                     "positive int. Write the chosen move into *out_pile and *out_remove. Scored by the fraction of "
                     "randomly-generated *winnable* positions (nonzero XOR of all piles) where the move actually wins "
                     "- not every position has a winning move, and there's no penalty for a merely-legal move on an "
                     "already-losing position.")}


@tool("nim", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C Nim-move kernel, check the fraction of random winnable positions where it finds a winning move
    (per Bouton's theorem), and time it. Returns value = speedup_vs_dp x fraction_correct.

    Args:
        source: Complete C source defining `void kernel(int n, const int *piles, int *out_pile, int *out_remove)`.
        sizes: Numbers of piles to test (default [3, 5, 8]).
        budget: Unused beyond compatibility; each size runs 300 fixed trials.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the nim tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [3, 5, 8])]
    if any(s < 1 or s > 64 for s in sizes):
        return {"error": "sizes (pile counts) must be between 1 and 64"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_nim_"))
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
        r["status"] = "exact" if r["frac_correct"] == 1.0 else "approx" if r["frac_correct"] > 0 else "WRONG"
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round((big["speedup_vs_dp"] or 0.0) * big["frac_correct"], 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "status": big["status"], "frac_correct": big["frac_correct"],
            "speedup_vs_dp": big["speedup_vs_dp"], "value": value, "prediction_target": big["speedup_vs_dp"],
            "reading": ("value = speedup vs. the XOR-rule reference x fraction of winnable positions where a "
                       "winning move was found - 1.0 fraction is Bouton-optimal, not just legal.")}
