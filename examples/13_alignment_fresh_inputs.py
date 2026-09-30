"""Check alignment kernels on fresh random inputs, not just the harness's fixed ones.

`alignment.bench` checks each length against ONE fixed sequence pair (seed 1234+n), so a kernel
that is not exact Needleman-Wunsch can still pass if it happens to get those pairs right.
This re-checks kernels on many new random pairs (and a few structured ones) against the same
reference DP.

    python examples/13_alignment_fresh_inputs.py <kernel.c> [<kernel.c> ...]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
int kernel(int n, const char *a, const char *b);
static int nw(int n,const char*a,const char*b,int*dp){
 for(int i=0;i<=n;i++)dp[i*(n+1)]=-2*i;for(int j=0;j<=n;j++)dp[j]=-2*j;
 for(int i=1;i<=n;i++)for(int j=1;j<=n;j++){
  int d=dp[(i-1)*(n+1)+j-1]+(a[i-1]==b[j-1]?1:-1),u=dp[(i-1)*(n+1)+j]-2,l=dp[i*(n+1)+j-1]-2;
  int m=d;if(u>m)m=u;if(l>m)m=l;dp[i*(n+1)+j]=m;}
 return dp[n*(n+1)+n];}
static uint64_t s=0x9E3779B97F4A7C15ULL;
static uint64_t r(void){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
int main(void){
 const char L[4]={'A','C','G','T'}; int sizes[]={7,33,64,100,257,513,1000,2048};
 int tested=0,wrong=0;
 for(int si=0;si<8;si++){int n=sizes[si];
  char*a=malloc(n+1),*b=malloc(n+1);int*dp=malloc((size_t)(n+1)*(n+1)*sizeof(int));
  for(int t=0;t<6;t++){
   for(int i=0;i<n;i++)a[i]=L[r()&3];
   if(t==4){for(int i=0;i<n;i++)b[i]=a[i];for(int k=0;k<n/10+1;k++)b[r()%n]=L[r()&3];}   /* similar pair */
   else if(t==5){for(int i=0;i<n;i++)b[i]='A';}                                          /* degenerate pair */
   else for(int i=0;i<n;i++)b[i]=L[r()&3];                                               /* independent random */
   a[n]=0;b[n]=0;
   int ref=nw(n,a,b,dp),got=kernel(n,a,b);tested++;
   if(ref!=got){wrong++;if(wrong<=5)printf("{\"mismatch\":{\"n\":%d,\"case\":%d,\"ref\":%d,\"got\":%d}}\n",n,t,ref,got);}}
  free(a);free(b);free(dp);}
 printf("{\"tested\":%d,\"wrong\":%d}\n",tested,wrong);return 0;}
"""


def check(kernel: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix="crazyai_alnfresh_"))
    (work / "h.c").write_text(HARNESS, encoding="utf-8")
    exe = work / "fresh.exe"
    cc = subprocess.run(["gcc", "-O2", "-march=native", "-fopenmp", "-o", str(exe), str(kernel), str(work / "h.c"), "-lm"],
                        capture_output=True, text=True, timeout=180)
    if cc.returncode != 0:
        return {"kernel": str(kernel), "error": "compile failed", "stderr": cc.stderr[-800:]}
    try:
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    except PermissionError as exc:
        return {"kernel": str(kernel), "error": f"blocked: {exc}"}
    finally:
        pass
    shutil.rmtree(work, ignore_errors=True)
    rows = [json.loads(l) for l in run.stdout.splitlines() if l.startswith("{")]
    out = {"kernel": str(kernel), "mismatches": [r["mismatch"] for r in rows if "mismatch" in r]}
    out.update(next((r for r in rows if "tested" in r), {"error": f"no result (exit {run.returncode})"}))
    out["exact_everywhere"] = out.get("wrong") == 0
    return out


if __name__ == "__main__":
    for k in sys.argv[1:]:
        print(json.dumps(check(Path(k))), flush=True)
