"""quantum.* - compile a C quantum-circuit statevector-simulation kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit,
               const double *gate_param, double *state_re, double *state_im);
    // apply n_gates single-qubit gates in sequence to a 2^n_qubits statevector.
    // gate_type: 0=Hadamard, 1=Pauli-X, 2=Pauli-Z, 3=Rz(gate_param). gate_qubit: target qubit.
    // state_re/state_im (length 2^n_qubits) arrive already initialised to |0...0>
    // (re[0]=1, else 0) and must be updated in place, gate by gate.

This is classical simulation of a quantum circuit (exactly what real
statevector simulators like Qiskit's Aer do under the hood) - fully
deterministic, fully measurable, no quantum hardware needed. The harness
(embedded below) generates a random gate sequence, checks the candidate's
final statevector against a reference (same gate-by-gate amplitude-pair
update) within a relative-error tolerance, and times both, the 3-tier-
exactness shape of kernel.py's/fft.py's harnesses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double "
           "*gate_param, double *state_re, double *state_im);  /* apply n_gates single-qubit gates in sequence to "
           "a 2^n_qubits statevector. gate_type: 0=Hadamard, 1=Pauli-X, 2=Pauli-Z, 3=Rz(gate_param[i]). "
           "gate_qubit: target qubit (0..n_qubits-1). state_re/state_im (length 2^n_qubits) arrive already "
           "initialised to |0...0> (re[0]=1, else 0) and must be updated in place, gate by gate */")

EXAMPLE = """#include <math.h>
static void apply_gate(int n, int gtype, int gq, double gparam, double *re, double *im) {
    long long bit = 1LL << gq, N = 1LL << n;
    for (long long i = 0; i < N; i++) {
        if (i & bit) continue;
        long long j = i | bit;
        double r0 = re[i], i0 = im[i], r1 = re[j], i1 = im[j];
        if (gtype == 0) {
            double s = 0.70710678118654752440;
            re[i] = (r0 + r1) * s; im[i] = (i0 + i1) * s; re[j] = (r0 - r1) * s; im[j] = (i0 - i1) * s;
        } else if (gtype == 1) {
            re[i] = r1; im[i] = i1; re[j] = r0; im[j] = i0;
        } else if (gtype == 2) {
            re[i] = r0; im[i] = i0; re[j] = -r1; im[j] = -i1;
        } else {
            double phi0 = -gparam / 2, phi1 = gparam / 2;
            double c0 = cos(phi0), s0 = sin(phi0), c1 = cos(phi1), s1 = sin(phi1);
            re[i] = r0 * c0 - i0 * s0; im[i] = r0 * s0 + i0 * c0;
            re[j] = r1 * c1 - i1 * s1; im[j] = r1 * s1 + i1 * c1;
        }
    }
}
void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
           double *state_re, double *state_im) {
    for (int g = 0; g < n_gates; g++) apply_gate(n_qubits, gate_type[g], gate_qubit[g], gate_param[g], state_re, state_im);
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
void kernel(int n_qubits,int n_gates,const int*gate_type,const int*gate_qubit,const double*gate_param,double*state_re,double*state_im);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void apply_gate(int n,int gtype,int gq,double gparam,double*re,double*im){
 long long bit=1LL<<gq,N=1LL<<n;
 for(long long i=0;i<N;i++){
  if(i&bit)continue;
  long long j=i|bit;
  double r0=re[i],i0=im[i],r1=re[j],i1=im[j];
  if(gtype==0){double s=0.70710678118654752440;re[i]=(r0+r1)*s;im[i]=(i0+i1)*s;re[j]=(r0-r1)*s;im[j]=(i0-i1)*s;}
  else if(gtype==1){re[i]=r1;im[i]=i1;re[j]=r0;im[j]=i0;}
  else if(gtype==2){re[i]=r0;im[i]=i0;re[j]=-r1;im[j]=-i1;}
  else{double phi0=-gparam/2,phi1=gparam/2;double c0=cos(phi0),s0=sin(phi0),c1=cos(phi1),s1=sin(phi1);
   re[i]=r0*c0-i0*s0;im[i]=r0*s0+i0*c0;re[j]=r1*c1-i1*s1;im[j]=r1*s1+i1*c1;}}}
static void ref_simulate(int nq,int ng,const int*gt,const int*gq,const double*gp,double*re,double*im){
 for(int g=0;g<ng;g++)apply_gate(nq,gt[g],gq[g],gp[g],re,im);}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
typedef void(*kfn)(int,int,const int*,const int*,const double*,double*,double*);
static double timeit(kfn f,int nq,int ng,const int*gt,const int*gq,const double*gp,double*re,double*im,long long N,double budget){
 re[0]=1;for(long long i=1;i<N;i++)re[i]=0;for(long long i=0;i<N;i++)im[i]=0;
 f(nq,ng,gt,gq,gp,re,im);
 double best=1e30,tot=0;int reps=0;
 while(reps<3||(tot<budget&&reps<40)){
  re[0]=1;for(long long i=1;i<N;i++)re[i]=0;for(long long i=0;i<N;i++)im[i]=0;
  double t0=now();f(nq,ng,gt,gq,gp,re,im);double dt=now()-t0;if(dt<best)best=dt;tot+=dt;reps++;}
 return best;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 for(int a=2;a<argc;a++){int nq=atoi(argv[a]);
  long long N=1LL<<nq;int ng=200;
  int*gt=malloc((size_t)ng*sizeof(int));int*gqb=malloc((size_t)ng*sizeof(int));double*gp=malloc((size_t)ng*sizeof(double));
  uint64_t s=5555+(uint64_t)nq;
  for(int g=0;g<ng;g++){gt[g]=(int)(xs(&s)%4);gqb[g]=(int)(xs(&s)%(uint64_t)nq);
   gp[g]=((double)(xs(&s)%1000)/1000.0)*6.28318530717958647692;}
  double*rref=malloc((size_t)N*8),*iref=malloc((size_t)N*8),*rout=malloc((size_t)N*8),*iout=malloc((size_t)N*8);
  double tref=timeit(ref_simulate,nq,ng,gt,gqb,gp,rref,iref,N,budget/2);
  double tk=timeit(kernel,nq,ng,gt,gqb,gp,rout,iout,N,budget);
  double num=0,den=0;int finite=1;
  for(long long i=0;i<N;i++){double dr=rout[i]-rref[i],di=iout[i]-iref[i];
   if(!isfinite(dr)||!isfinite(di)){finite=0;break;}
   num+=dr*dr+di*di;den+=rref[i]*rref[i]+iref[i]*iref[i];}
  double rel=finite?sqrt(num/(den>0?den:1)):INFINITY;
  printf("{\"n_qubits\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"rel_err\":%.3e}\n",nq,tk,tref,rel);
  fflush(stdout);free(gt);free(gqb);free(gp);free(rref);free(iref);free(rout);free(iout);}
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


@tool("quantum", "measure")
def contract() -> dict:
    """The fixed C signature every quantum-simulation idea must implement, an example (gate-by-gate) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": "n_qubits is small (harness uses 8, 12, 16 - up to 65536 amplitudes). 200 random gates per "
                     "run. state_re/state_im arrive as |0...0> - the kernel must not re-initialise them."}


@tool("quantum", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C quantum-circuit-simulation kernel, check its final statevector against a reference gate-by-gate
    simulator and time it. Returns error, speedup vs. the reference, and value = speedup x exactness.

    Args:
        source: Complete C source defining the kernel per CONTRACT.
        sizes: Qubit counts to run (default [8, 12, 16] -> up to 65536 amplitudes).
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the quantum tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [8, 12, 16])]
    if any(s < 1 or s > 20 for s in sizes):
        return {"error": "sizes (qubit counts) must be between 1 and 20"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_quantum_"))
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
    big = max(rows, key=lambda r: r["n_qubits"])
    value = round((big["speedup_vs_naive"] or 0.0) * big["exactness"], 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n_qubits"], "rel_err": big["rel_err"], "status": big["status"],
            "speedup_vs_naive": big["speedup_vs_naive"], "value": value, "prediction_target": big["speedup_vs_naive"],
            "reading": "value = speedup vs. a reference gate-by-gate statevector simulator x exactness (1.0 relative error < 1e-9)."}
