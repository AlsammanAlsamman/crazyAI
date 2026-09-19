"""rs_encode.* - compile a C Reed-Solomon encoding kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int k, int n, const unsigned char *data, unsigned char *parity);
    // systematic Reed-Solomon encoding over GF(2^8) (reducing polynomial 0x11D,
    // primitive element 2): k message bytes in, n-k parity bytes out.

Scoped to encoding only (not full decode/error-correction, which needs
Berlekamp-Massey/Chien search - real but substantially more intricate).
The harness (embedded below) checks the kernel's parity bytes against a
reference systematic encoder (generator polynomial built from repeated
multiplication by (x - alpha^i), synthetic division via a naive shift-
and-reduce GF(2^8) multiply) - exact byte match, no tolerance, encoding is
deterministic - and times both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("void kernel(int k, int n, const unsigned char *data, unsigned char *parity);  /* systematic "
           "Reed-Solomon encoding over GF(2^8) (reducing polynomial 0x11D, primitive element 2): data has k "
           "message bytes, parity (length n-k) is filled with the parity bytes; n-k is fixed at 16 by the "
           "harness */")

EXAMPLE = """#include <stdlib.h>
#include <string.h>
static unsigned char gf_mul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) p ^= a;
        int hi = a & 0x80;
        a = (unsigned char)(a << 1);
        if (hi) a ^= 0x1D;
        b >>= 1;
    }
    return p;
}
static unsigned char gf_pow(unsigned char a, int e) { unsigned char r = 1; for (int i = 0; i < e; i++) r = gf_mul(r, a); return r; }
static void gen_poly(int nsym, unsigned char *gen) {
    gen[0] = 1;
    for (int i = 1; i <= nsym; i++) gen[i] = 0;
    int glen = 1;
    for (int i = 0; i < nsym; i++) {
        unsigned char root = gf_pow(2, i);
        unsigned char newgen[64];
        for (int j = 0; j <= glen; j++) newgen[j] = 0;
        for (int j = 0; j < glen; j++) { newgen[j] ^= gen[j]; newgen[j + 1] ^= gf_mul(gen[j], root); }
        glen++;
        for (int j = 0; j < glen; j++) gen[j] = newgen[j];
    }
}
void kernel(int k, int n, const unsigned char *data, unsigned char *parity) {
    int nsym = n - k;
    unsigned char gen[64];
    gen_poly(nsym, gen);
    unsigned char *msg = calloc((size_t)(k + nsym), 1);
    memcpy(msg, data, (size_t)k);
    for (int i = 0; i < k; i++) {
        unsigned char coef = msg[i];
        if (coef != 0) for (int j = 0; j <= nsym; j++) if (gen[j]) msg[i + j] ^= gf_mul(gen[j], coef);
    }
    memcpy(parity, msg + k, (size_t)nsym);
    free(msg);
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
void kernel(int k,int n,const unsigned char*data,unsigned char*parity);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static unsigned char gf_mul(unsigned char a,unsigned char b){
 unsigned char p=0;
 for(int i=0;i<8;i++){if(b&1)p^=a;int hi=a&0x80;a=(unsigned char)(a<<1);if(hi)a^=0x1D;b>>=1;}
 return p;}
static unsigned char gf_pow(unsigned char a,int e){unsigned char r=1;for(int i=0;i<e;i++)r=gf_mul(r,a);return r;}
static void gen_poly(int nsym,unsigned char*gen){
 gen[0]=1;for(int i=1;i<=nsym;i++)gen[i]=0;
 int glen=1;
 for(int i=0;i<nsym;i++){unsigned char root=gf_pow(2,i);unsigned char newgen[64];
  for(int j=0;j<=glen;j++)newgen[j]=0;
  for(int j=0;j<glen;j++){newgen[j]^=gen[j];newgen[j+1]^=gf_mul(gen[j],root);}
  glen++;for(int j=0;j<glen;j++)gen[j]=newgen[j];}}
static void ref_rs_encode(int k,int n,const unsigned char*data,unsigned char*parity){
 int nsym=n-k;unsigned char gen[64];gen_poly(nsym,gen);
 unsigned char*msg=calloc((size_t)(k+nsym),1);
 memcpy(msg,data,(size_t)k);
 for(int i=0;i<k;i++){unsigned char coef=msg[i];
  if(coef)for(int j=0;j<=nsym;j++)if(gen[j])msg[i+j]^=gf_mul(gen[j],coef);}
 memcpy(parity,msg+k,(size_t)nsym);free(msg);}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
typedef void(*kfn)(int,int,const unsigned char*,unsigned char*);
static double timeit(kfn f,int k,int n,const unsigned char*data,unsigned char*parity,double budget){
 f(k,n,data,parity);double best=1e30,tot=0;int reps=0;
 while(reps<3||(tot<budget&&reps<40)){
  double t0=now();f(k,n,data,parity);double dt=now()-t0;if(dt<best)best=dt;tot+=dt;reps++;}
 return best;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 int NSYM=16;
 for(int a=2;a<argc;a++){int k=atoi(argv[a]);int n=k+NSYM;
  unsigned char*data=malloc((size_t)k);uint64_t s=2024+(uint64_t)k;
  for(int i=0;i<k;i++)data[i]=(unsigned char)(xs(&s)&0xFF);
  unsigned char*pref=malloc((size_t)NSYM),*pout=malloc((size_t)NSYM);
  double tref=timeit(ref_rs_encode,k,n,data,pref,budget/2);
  double tk=timeit(kernel,k,n,data,pout,budget);
  int mismatches=0;
  for(int i=0;i<NSYM;i++)if(pref[i]!=pout[i])mismatches++;
  printf("{\"k\":%d,\"n\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"mismatches\":%d}\n",k,n,tk,tref,mismatches);
  fflush(stdout);free(data);free(pref);free(pout);}
 return 0;}
"""


@tool("rs_encode", "measure")
def contract() -> dict:
    """The fixed C signature every Reed-Solomon-encoding idea must implement, an example (naive LFSR) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp",
            "notes": "n-k (the number of parity bytes) is fixed at 16 by the harness. GF(2^8) uses reducing "
                     "polynomial 0x11D and primitive element 2. data is random bytes."}


@tool("rs_encode", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C Reed-Solomon-encoding kernel, check its parity bytes against a reference LFSR encoder (exact
    match) and time it. Returns error, speedup vs. the reference, and value = speedup x (1.0 if exact else 0.0).

    Args:
        source: Complete C source defining `void kernel(int k, int n, const unsigned char *data, unsigned char *parity)`.
        sizes: Message lengths (k, in bytes) to run (default [1024, 8192, 65536]); n = k + 16 in every case.
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the rs_encode tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [1024, 8192, 65536])]
    if any(s < 16 or s > 2_000_000 for s in sizes):
        return {"error": "sizes must be between 16 and 2,000,000"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_rs_"))
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
    big = max(rows, key=lambda r: r["k"])
    value = round((big["speedup_vs_naive"] or 0.0) * (1.0 if big["correct"] else 0.0), 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["k"], "status": big["status"],
            "speedup_vs_naive": big["speedup_vs_naive"], "value": value, "prediction_target": big["speedup_vs_naive"],
            "reading": "value = speedup vs. a reference naive shift-and-reduce LFSR encoder, 0 if the parity bytes don't match exactly."}
