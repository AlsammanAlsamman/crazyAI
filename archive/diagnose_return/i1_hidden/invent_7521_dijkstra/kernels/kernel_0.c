/* Single-source shortest distances, weighted directed graph, weights >= 0.
   Literal translation of SEED 1: a flock lands on a whole BAND of smallest
   letters at once; stones are cast from while still tentative; dead threads
   are dropped; a locked letter is never rebuilt.
   Guarded fallback/bailout: lazy 4-ary heap ("one bird, one stone").
   gcc -O3 -march=native -fopenmp -lm                                       */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* A garden-stone letter is the IEEE-754 bit pattern of its distance.  For
   non-negative doubles (and +INF) unsigned order == numeric order, so every
   "is this letter higher?" test is one integer compare and the final answer
   is one memcpy.  Keeping the working array as uint64_t also makes the
   atomic CAS-min strictly legal (no type punning of double*).            */
#define GS_INF 0x7FF0000000000000ULL

static inline double  L2D(uint64_t k){ double d; memcpy(&d,&k,sizeof d); return d; }
static inline uint64_t D2L(double d){ uint64_t k; memcpy(&k,&d,sizeof k); return k; }

static inline long long band_of(double d, double invD){
    double q = d*invD;
    if(!(q < 9.0e18)) return LLONG_MAX;      /* astronomically far / +inf */
    return (long long)q;
}

/* ---------------- safe path: lazy 4-ary heap ---------------- */
typedef struct { uint64_t k; int v; } HN;
typedef struct { HN *a; long long n, cap; } Heap;

static void hp_push(Heap *h, uint64_t k, int v){
    if(h->n == h->cap){
        long long c = h->cap ? h->cap*2 : 4096;
        HN *na = (HN*)realloc(h->a, (size_t)c*sizeof(HN));
        if(!na) return;
        h->a = na; h->cap = c;
    }
    HN *a = h->a; long long i = h->n++;
    while(i > 0){ long long p=(i-1)>>2; if(a[p].k <= k) break; a[i]=a[p]; i=p; }
    a[i].k = k; a[i].v = v;
}
static HN hp_pop(Heap *h){
    HN *a = h->a, top = a[0];
    long long nn = --h->n;
    if(nn > 0){
        HN last = a[nn]; long long i = 0;
        for(;;){
            long long c = (i<<2)+1; if(c >= nn) break;
            long long e = c+4; if(e > nn) e = nn;
            long long b = c; uint64_t bk = a[c].k;
            for(long long j=c+1;j<e;j++) if(a[j].k < bk){ bk=a[j].k; b=j; }
            if(bk >= last.k) break;
            a[i] = a[b]; i = b;
        }
        a[i] = last;
    }
    return top;
}
static void run_heap(const int *rs,const int *adj,const double *wt,
                     uint64_t *key, Heap *h){
    while(h->n > 0){
        HN t = hp_pop(h);
        int u = t.v;
        if(t.k > key[u]) continue;           /* thread into bramble: dropped */
        double du = L2D(t.k);
        int e = rs[u], ee = rs[u+1];
        for(; e < ee; e++){
            int v = adj[e];
            uint64_t nk = D2L(du + wt[e]);
            if(nk < key[v]){ key[v] = nk; hp_push(h, nk, v); }
        }
    }
}

/* ---------------- bands (bags of stones) ---------------- */
typedef struct { int *a; int n, cap; } Bag;
static inline void bag_push(Bag *b, int x){
    if(b->n == b->cap){
        int c = b->cap ? b->cap*2 : 16;
        int *na = (int*)realloc(b->a, (size_t)c*sizeof(int));
        if(!na) return;
        b->a = na; b->cap = c;
    }
    b->a[b->n++] = x;
}
typedef struct { long long ix; int v; } PEnt;
typedef struct { PEnt *a; long n, cap; } PBag;
static inline void pbag_push(PBag *b, long long ix, int v){
    if(b->n == b->cap){
        long c = b->cap ? b->cap*2 : 1024;
        PEnt *na = (PEnt*)realloc(b->a, (size_t)c*sizeof(PEnt));
        if(!na) return;
        b->a = na; b->cap = c;
    }
    b->a[b->n].ix = ix; b->a[b->n].v = v; b->n++;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if(n <= 0 || !dist_out) return;
    if(m <= 0 || (unsigned)source >= (unsigned)n){
        for(int i=0;i<n;i++) dist_out[i] = INFINITY;
        if((unsigned)source < (unsigned)n) dist_out[source] = 0.0;
        return;
    }

    /* ---- SEED 2: roads, directional, priced by their thief ---- */
    int    *rs  = (int*)   calloc((size_t)n+1, sizeof(int));
    int    *adj = (int*)   malloc((size_t)m*sizeof(int));
    double *wt  = (double*)malloc((size_t)m*sizeof(double));
    int    *cur = (int*)   malloc((size_t)n*sizeof(int));
    uint64_t *key = (uint64_t*)malloc((size_t)n*sizeof(uint64_t));
    uint64_t *sk  = (uint64_t*)malloc((size_t)n*sizeof(uint64_t));
    if(!rs||!adj||!wt||!cur||!key||!sk){
        for(int i=0;i<n;i++) dist_out[i] = INFINITY;
        dist_out[source] = 0.0;
        free(rs);free(adj);free(wt);free(cur);free(key);free(sk);
        return;
    }

    double wmax = 0.0, wsum = 0.0; int bad = 0;
    for(int i=0;i<m;i++){
        int s = src[i], t = dst[i];
        if((unsigned)s >= (unsigned)n || (unsigned)t >= (unsigned)n) continue;
        double x = weight[i];
        if(!(x >= 0.0) || !(x <= 1.7976931348623157e308)) bad = 1; /* neg/NaN/inf */
        if(x > wmax) wmax = x;
        wsum += x;
        rs[s+1]++;
    }
    for(int i=0;i<n;i++) rs[i+1] += rs[i];
    int me = rs[n];
    memcpy(cur, rs, (size_t)n*sizeof(int));
    for(int i=0;i<m;i++){
        int s = src[i], t = dst[i];
        if((unsigned)s >= (unsigned)n || (unsigned)t >= (unsigned)n) continue;
        int p = cur[s]++; adj[p] = t; wt[p] = weight[i];
    }
    free(cur);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 200000)
#endif
    for(int i=0;i<n;i++) key[i] = GS_INF;      /* every stone: unreadable debt */
    memset(sk, 0xFF, (size_t)n*sizeof(uint64_t));
    key[source] = 0;                            /* "nothing owed" underfoot   */

    /* ---- band width, and the guards that decide which mechanism runs ---- */
    double delta = (me > 0) ? wsum/(double)me : 0.0;   /* mean toll */
    int use_bands = 1;
    if(bad)                                  use_bands = 0; /* weird weights */
    if(wmax <= 0.0)                          use_bands = 0; /* all tolls 0   */
    if(!(delta > 0.0) || !(delta < 1e300))   use_bands = 0;
    if(me < 2048 || n < 64)                  use_bands = 0; /* too small     */
    if(use_bands && wmax/delta > 1.0e6)      use_bands = 0; /* toll spread    */

    Heap h = {NULL,0,0};

    if(!use_bands){
        hp_push(&h, 0, source);
        run_heap(rs, adj, wt, key, &h);
    } else {
        int B = 1<<14;
        if(n < B){ B = 256; while(B < n) B <<= 1; }
        Bag *ring = (Bag*)calloc((size_t)B, sizeof(Bag));
        if(!ring){
            hp_push(&h, 0, source);
            run_heap(rs, adj, wt, key, &h);
        } else {
            Bag ovf = {NULL,0,0}, front = {NULL,0,0};
            double invD = 1.0/delta;
            long long base = 0;
            int slot = 0, bail = 0;
            long long work = 0, limit = 8LL*((long long)me + (long long)n) + 4096;
            double avgdeg = (double)me/(double)n;
            int T = 1;
            PBag *pb = NULL;
#ifdef _OPENMP
            T = omp_get_max_threads();
            if(T < 1) T = 1; if(T > 64) T = 64;
            if(T > 1){ pb = (PBag*)calloc((size_t)T, sizeof(PBag)); if(!pb) T = 1; }
#endif
            bag_push(&ring[0], source);

            for(;;){
                if(slot >= B){                       /* window exhausted: refill */
                    if(ovf.n == 0) break;
                    long long mn = LLONG_MAX;
                    for(int j=0;j<ovf.n;j++){
                        long long ix = band_of(L2D(key[ovf.a[j]]), invD);
                        if(ix < mn) mn = ix;
                    }
                    base = mn; slot = 0;
                    int keep = 0;
                    for(int j=0;j<ovf.n;j++){
                        int u = ovf.a[j];
                        long long ix = band_of(L2D(key[u]), invD) - base;
                        if(ix < 0) ix = 0;
                        if(ix < (long long)B) bag_push(&ring[(int)ix], u);
                        else ovf.a[keep++] = u;
                    }
                    ovf.n = keep;
                    continue;
                }
                if(ring[slot].n == 0){ slot++; continue; }

                long long gslot = base + slot;
                /* the flock lands on this whole band and keeps landing on it
                   until nothing in it moves: stones here are cast from while
                   still tentative, detached and called back on.            */
                while(ring[slot].n > 0){
                    { Bag tmp = ring[slot]; ring[slot] = front; front = tmp; }
                    ring[slot].n = 0;
                    int  fn = front.n;
                    int *fa = front.a;
                    int par = 0;
#ifdef _OPENMP
                    if(T > 1 && (double)fn*avgdeg >= 30000.0) par = 1;
#endif
                    if(!par){
                        for(int t=0;t<fn;t++){
                            int u = fa[t];
                            uint64_t ku = key[u];
                            if(band_of(L2D(ku), invD) != gslot) continue;  /* dropped */
                            if(sk[u] <= ku) continue;                      /* dropped */
                            sk[u] = ku;
                            double du = L2D(ku);
                            int e = rs[u], ee = rs[u+1];
                            work += ee - e;
                            for(; e < ee; e++){
                                if(e+4 < ee) __builtin_prefetch(&key[adj[e+4]], 1, 1);
                                int v = adj[e];
                                double nd = du + wt[e];
                                uint64_t nk = D2L(nd);
                                if(nk < key[v]){
                                    key[v] = nk;
                                    long long ix = band_of(nd, invD) - base;
                                    if(ix < 0) ix = 0;
                                    if(ix < (long long)B) bag_push(&ring[(int)ix], v);
                                    else bag_push(&ovf, v);
                                }
                            }
                        }
                    }
#ifdef _OPENMP
                    else {
                        long long wl = 0;
                        #pragma omp parallel num_threads(T) reduction(+:wl)
                        {
                            int tid = omp_get_thread_num();
                            PBag *b = &pb[tid]; b->n = 0;
                            #pragma omp for schedule(dynamic,64)
                            for(int t=0;t<fn;t++){
                                int u = fa[t];
                                uint64_t ku = __atomic_load_n(&key[u], __ATOMIC_RELAXED);
                                if(band_of(L2D(ku), invD) != gslot) continue;
                                uint64_t prev = __atomic_exchange_n(&sk[u], ku,
                                                                   __ATOMIC_RELAXED);
                                if(prev <= ku) continue;
                                double du = L2D(ku);
                                int e = rs[u], ee = rs[u+1];
                                wl += ee - e;
                                for(; e < ee; e++){
                                    int v = adj[e];
                                    double nd = du + wt[e];
                                    uint64_t nk = D2L(nd);
                                    uint64_t old = __atomic_load_n(&key[v],
                                                                   __ATOMIC_RELAXED);
                                    while(nk < old){
                                        if(__atomic_compare_exchange_n(&key[v], &old, nk,
                                                0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                                            pbag_push(b, band_of(nd, invD), v);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        work += wl;
                        for(int q=0;q<T;q++){
                            PBag *b = &pb[q];
                            for(long j=0;j<b->n;j++){
                                long long ix = b->a[j].ix - base;
                                int v = b->a[j].v;
                                if(ix < 0) ix = 0;
                                if(ix < (long long)B) bag_push(&ring[(int)ix], v);
                                else bag_push(&ovf, v);
                            }
                            b->n = 0;
                        }
                    }
#endif
                    if(work > limit){ bail = 1; break; }
                }
                if(bail) break;
                slot++;
            }

            if(bail){
                /* Band width was a bad bet for this graph: every stone whose
                   letter can still fall is queued, so hand the whole queue to
                   the one-bird-one-stone heap and finish there.  Exact.     */
                for(int s2=slot; s2<B; s2++){
                    for(int j=0;j<ring[s2].n;j++){
                        int u = ring[s2].a[j];
                        hp_push(&h, key[u], u);
                    }
                    ring[s2].n = 0;
                }
                for(int j=0;j<ovf.n;j++){
                    int u = ovf.a[j];
                    hp_push(&h, key[u], u);
                }
                ovf.n = 0;
                run_heap(rs, adj, wt, key, &h);
            }

            for(int s2=0;s2<B;s2++) free(ring[s2].a);
            free(ring); free(ovf.a); free(front.a);
            if(pb){ for(int q=0;q<T;q++) free(pb[q].a); free(pb); }
        }
    }

    memcpy(dist_out, key, (size_t)n*sizeof(double));   /* letters -> answers */

    free(h.a); free(rs); free(adj); free(wt); free(key); free(sk);
}
