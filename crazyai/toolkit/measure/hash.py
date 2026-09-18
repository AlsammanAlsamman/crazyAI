"""hash.* - compile a C hash function, check how well it mixes, time it.

The contract is fixed so that every idea is measured on the same ground:

    uint64_t kernel(const unsigned char *data, size_t len);

The harness (embedded below) measures throughput vs. an FNV-1a reference and
an avalanche score: for many random buffers, flip one random input bit and
report how close the fraction of output bits that changed is to 0.5 (ideal
mixing). There is no "correct" hash to check against - only good or bad
mixing - so this tool's `value` is speed x mixing quality, not speed x
exactness.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = "uint64_t kernel(const unsigned char *data, size_t len);  /* hash len bytes of data to a 64-bit value */"

EXAMPLE = """#include <stdint.h>
uint64_t kernel(const unsigned char *data, size_t len) {
    uint64_t h = 1469598103934665603ULL;   /* FNV-1a offset basis */
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;             /* FNV prime */
    }
    return h;
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
uint64_t kernel(const unsigned char *data, size_t len);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
static uint64_t fnv1a(const unsigned char*data,size_t len){uint64_t h=1469598103934665603ULL;
 for(size_t i=0;i<len;i++){h^=data[i];h*=1099511628211ULL;}return h;}
#define AVAL_TRIALS 200
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;(void)budget;
 for(int ai=2;ai<argc;ai++){int len=atoi(argv[ai]);
  unsigned char*buf=malloc(len),*buf2=malloc(len);uint64_t s=1234+len;
  for(int i=0;i<len;i++)buf[i]=(unsigned char)(xs(&s)&0xFF);
  int BATCH=len<64?200000:(len<4096?20000:2000);
  volatile uint64_t sink=0;double t0=now();for(int b=0;b<BATCH;b++)sink+=kernel(buf,(size_t)len);double tk=(now()-t0)/BATCH;
  volatile uint64_t sink2=0;double t1=now();for(int b=0;b<BATCH;b++)sink2+=fnv1a(buf,(size_t)len);double tref=(now()-t1)/BATCH;
  double avg_frac=0;
  for(int trial=0;trial<AVAL_TRIALS;trial++){
   for(int i=0;i<len;i++)buf[i]=(unsigned char)(xs(&s)&0xFF);
   memcpy(buf2,buf,len);
   uint64_t bitpos=xs(&s)%((uint64_t)len*8);
   buf2[bitpos/8]^=(1<<(bitpos%8));
   uint64_t h1=kernel(buf,(size_t)len),h2=kernel(buf2,(size_t)len);
   int popcount=__builtin_popcountll(h1^h2);
   avg_frac+=(double)popcount/64.0;}
  avg_frac/=AVAL_TRIALS;
  double avalanche_score=1.0-2.0*fabs(0.5-avg_frac);
  printf("{\"n\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"avg_avalanche_frac\":%.4f,\"avalanche_score\":%.4f}\n",
   len,tk,tref,avg_frac,avalanche_score);fflush(stdout);free(buf);free(buf2);}
 return 0;}
"""


@tool("hash", "measure")
def contract() -> dict:
    """The fixed C signature every hash-function idea must implement, an example (FNV-1a) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": ("data/len describe one byte buffer; return one 64-bit value. Scored on throughput vs. FNV-1a "
                     "and an avalanche score (1.0 = flipping one input bit flips ~50% of output bits, the ideal; "
                     "0.0 = flipping a bit changes nothing or everything). There is no 'correct' hash to match.")}


@tool("hash", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C hash kernel, measure its avalanche (bit-mixing) quality and throughput vs. FNV-1a.
    Returns value = speedup_vs_dp x avalanche_score.

    Args:
        source: Complete C source defining `uint64_t kernel(const unsigned char *data, size_t len)`.
        sizes: Buffer lengths in bytes to test (default [16, 256, 4096]).
        budget: Unused beyond compatibility; each size runs a fixed batch/trial count.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the hash tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [16, 256, 4096])]
    if any(s < 8 or s > 65536 for s in sizes):
        return {"error": "sizes (buffer bytes) must be between 8 and 65536"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_hash_"))
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
        r["status"] = ("exact" if r["avalanche_score"] > 0.9 else "approx" if r["avalanche_score"] > 0.3 else "WRONG")
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round((big["speedup_vs_dp"] or 0.0) * big["avalanche_score"], 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "status": big["status"], "avalanche_score": big["avalanche_score"],
            "speedup_vs_dp": big["speedup_vs_dp"], "value": value, "prediction_target": big["speedup_vs_dp"],
            "reading": ("value = speedup vs. FNV-1a x avalanche_score (1.0 = ideal ~50% output-bit flip rate per "
                       "input-bit flip). 'status' here means mixing quality, not correctness - there is no reference "
                       "hash to match exactly.")}
