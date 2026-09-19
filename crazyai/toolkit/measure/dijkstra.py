"""dijkstra.* - compile a C shortest-path kernel, check it, time it.

The contract is fixed so that every idea is measured on the same ground:

    void kernel(int n, int m, const int *src, const int *dst, const double *weight,
               int source, double *dist_out);
    // single-source shortest distances on a weighted directed graph (n nodes,
    // m directed edges src[i]->dst[i] weight weight[i], all weights >= 0);
    // dist_out (length n) filled with the shortest distance from source,
    // INFINITY for an unreachable node.

The harness (embedded below) generates a random directed graph (~4n edges,
random non-negative weights), checks the candidate against a reference
binary-heap Dijkstra (matching reachability exactly, and distances within a
relative-error tolerance - floating point summation order can differ between
an equivalent-but-differently-shaped algorithm), and times both, the same
3-tier-exactness shape as kernel.py's/fft.py's harnesses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crazyai.toolkit.registry import tool

CONTRACT = ("void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, "
           "double *dist_out);  /* single-source shortest distances on a weighted directed graph (n nodes, m "
           "directed edges src[i]->dst[i] weight weight[i], all weights >= 0); dist_out (length n) filled with the "
           "shortest distance from source, INFINITY for an unreachable node */")

EXAMPLE = """#include <stdlib.h>
#include <math.h>
typedef struct { double d; int u; } HeapItem;
static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break; HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}
void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) { int u = src[i]; int pos = off[u] + fill[u]++; edst[pos] = dst[i]; ew[pos] = weight[i]; }
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0;
    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0, source);
    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = dist_out[u] + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }
    free(deg); free(off); free(edst); free(ew); free(fill); free(done); free(heap);
}
"""

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
void kernel(int n,int m,const int*src,const int*dst,const double*weight,int source,double*dist_out);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct{double d;int u;}HeapItem;
static void hpush(HeapItem*h,int*hs,double d,int u){int i=(*hs)++;h[i].d=d;h[i].u=u;
 while(i>0){int p=(i-1)/2;if(h[p].d<=h[i].d)break;HeapItem t=h[p];h[p]=h[i];h[i]=t;i=p;}}
static HeapItem hpop(HeapItem*h,int*hs){HeapItem top=h[0];(*hs)--;h[0]=h[*hs];
 int i=0;while(1){int l=2*i+1,r=2*i+2,s=i;
  if(l<*hs&&h[l].d<h[s].d)s=l;if(r<*hs&&h[r].d<h[s].d)s=r;
  if(s==i)break;HeapItem t=h[s];h[s]=h[i];h[i]=t;i=s;}return top;}
static void ref_dijkstra(int n,int m,const int*src,const int*dst,const double*weight,int source,double*dist){
 int*deg=calloc((size_t)n,sizeof(int));
 for(int i=0;i<m;i++)deg[src[i]]++;
 int*off=malloc((size_t)(n+1)*sizeof(int));off[0]=0;
 for(int i=0;i<n;i++)off[i+1]=off[i]+deg[i];
 int*edst=malloc((size_t)m*sizeof(int));double*ew=malloc((size_t)m*sizeof(double));
 int*fill=calloc((size_t)n,sizeof(int));
 for(int i=0;i<m;i++){int u=src[i];int pos=off[u]+fill[u]++;edst[pos]=dst[i];ew[pos]=weight[i];}
 for(int i=0;i<n;i++)dist[i]=INFINITY;
 dist[source]=0;
 char*done=calloc((size_t)n,1);
 HeapItem*heap=malloc((size_t)(m+2)*sizeof(HeapItem));int hs=0;
 hpush(heap,&hs,0,source);
 while(hs>0){HeapItem top=hpop(heap,&hs);int u=top.u;if(done[u])continue;done[u]=1;
  for(int e=off[u];e<off[u+1];e++){int v=edst[e];double nd=dist[u]+ew[e];
   if(nd<dist[v]){dist[v]=nd;hpush(heap,&hs,nd,v);}}}
 free(deg);free(off);free(edst);free(ew);free(fill);free(done);free(heap);}
static uint64_t xs(uint64_t*s){uint64_t x=*s;x^=x>>12;x^=x<<25;x^=x>>27;*s=x;return x*0x2545F4914F6CDD1DULL;}
typedef void(*kfn)(int,int,const int*,const int*,const double*,int,double*);
static double timeit(kfn f,int n,int m,const int*src,const int*dst,const double*w,int source,double*dist,double budget){
 f(n,m,src,dst,w,source,dist);
 double best=1e30,tot=0;int reps=0;
 while(reps<3||(tot<budget&&reps<40)){
  double t0=now();f(n,m,src,dst,w,source,dist);double dt=now()-t0;if(dt<best)best=dt;tot+=dt;reps++;}
 return best;}
int main(int argc,char**argv){double budget=argc>1?atof(argv[1]):0.3;
 for(int a=2;a<argc;a++){int n=atoi(argv[a]);
  int m=n*4;if(m<4)m=4;
  int*src=malloc((size_t)m*sizeof(int));int*dst=malloc((size_t)m*sizeof(int));double*w=malloc((size_t)m*sizeof(double));
  uint64_t s=777+(uint64_t)n;
  for(int i=0;i<m;i++){src[i]=(int)(xs(&s)%(uint64_t)n);
   int d=(int)(xs(&s)%(uint64_t)n);while(d==src[i])d=(int)(xs(&s)%(uint64_t)n);
   dst[i]=d;w[i]=0.1+(double)(xs(&s)%1000)/100.0;}
  double*dref=malloc((size_t)n*sizeof(double));double*dout=malloc((size_t)n*sizeof(double));
  double tref=timeit(ref_dijkstra,n,m,src,dst,w,0,dref,budget/2);
  double tk=timeit(kernel,n,m,src,dst,w,0,dout,budget);
  int reach_mismatch=0;double max_rel=0;
  for(int i=0;i<n;i++){int ri=isinf(dref[i]),oi=isinf(dout[i]);
   if(ri!=oi){reach_mismatch++;continue;}
   if(!ri){double diff=fabs(dout[i]-dref[i]);double denom=fabs(dref[i])>1?fabs(dref[i]):1;double rel=diff/denom;if(rel>max_rel)max_rel=rel;}}
  printf("{\"n\":%d,\"m\":%d,\"time\":%.9g,\"ref_time\":%.9g,\"reach_mismatch\":%d,\"max_rel_err\":%.3e}\n",
        n,m,tk,tref,reach_mismatch,max_rel);
  fflush(stdout);
  free(src);free(dst);free(w);free(dref);free(dout);}
 return 0;}
"""


def _exactness(reach_mismatch: int, rel: float) -> float:
    if reach_mismatch > 0:
        return 0.0
    if rel < 1e-9:
        return 1.0
    if rel < 1e-6:
        return 0.5
    if rel < 1e-3:
        return 0.1
    return 0.0


@tool("dijkstra", "measure")
def contract() -> dict:
    """The fixed C signature every shortest-path idea must implement, an example (heap Dijkstra) kernel, and the compiler flags used."""
    return {"signature": CONTRACT, "example": EXAMPLE, "flags": "gcc -O3 -march=native -fopenmp -lm",
            "notes": "The harness generates a random directed graph with ~4n edges and non-negative random weights "
                     "for each size n. source is always node 0."}


@tool("dijkstra", "measure")
def bench(source: str, sizes: list | None = None, budget: float = 0.3, ctx: dict | None = None) -> dict:
    """Compile a C shortest-path kernel, check it against a reference binary-heap Dijkstra and time it. Returns
    error, speedup vs. the reference, and value = speedup x exactness.

    Args:
        source: Complete C source defining `void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out)`.
        sizes: Node counts to run, each generating ~4x as many random edges (default [200, 1000, 5000]).
        budget: Seconds of repeats per size for the kernel.
    """
    if shutil.which("gcc") is None:
        return {"error": "gcc not found; the dijkstra tool needs a C compiler"}
    sizes = [int(s) for s in (sizes or [200, 1000, 5000])]
    if any(s < 4 or s > 200_000 for s in sizes):
        return {"error": "sizes must be between 4 and 200,000"}
    run_dir = Path((ctx or {}).get("run_dir", "")) if (ctx or {}).get("run_dir") else None
    work = Path(tempfile.mkdtemp(prefix="crazyai_dijkstra_"))
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
        r["speedup_vs_heap"] = round(r["ref_time"] / r["time"], 3) if r["ref_time"] > 0 else None
        r["exactness"] = _exactness(r["reach_mismatch"], r["max_rel_err"])
        r["status"] = "exact" if r["exactness"] == 1.0 else "approx" if r["exactness"] > 0 else "WRONG"
    if not rows:
        return {"error": "no results", "stdout": run.stdout[-1000:], "stderr": run.stderr[-1000:]}
    big = max(rows, key=lambda r: r["n"])
    value = round((big["speedup_vs_heap"] or 0.0) * big["exactness"], 4)
    if run_dir:
        kd = run_dir / "kernels"
        kd.mkdir(parents=True, exist_ok=True)
        i = len(list(kd.glob("*.c")))
        (kd / f"kernel_{i}.c").write_text(source, encoding="utf-8")
        (kd / f"kernel_{i}.json").write_text(json.dumps({"results": rows, "value": value}, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return {"results": rows, "largest_n": big["n"], "status": big["status"],
            "speedup_vs_heap": big["speedup_vs_heap"], "value": value, "prediction_target": big["speedup_vs_heap"],
            "reading": "value = speedup vs. a reference binary-heap Dijkstra x exactness (1.0 = every node's "
                       "reachability and distance matches to within 1e-9 relative error)."}
