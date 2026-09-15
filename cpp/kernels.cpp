// crazyAI native kernels.
//
// Small, dependency-free numeric routines that are called many times inside
// the measure tools (sieves, orbit computations, Monte Carlo). Exposed with a
// C ABI so Python can load them through ctypes. Every routine has a pure
// Python fallback in crazyai/toolkit/native.py; this file only makes things
// faster, never possible.

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

extern "C" {

// Sieve of Eratosthenes. out[i] = 1 if i is prime, else 0. out has n+1 slots.
void sieve(uint64_t n, uint8_t* out) {
    std::memset(out, 1, n + 1);
    if (n >= 0) out[0] = 0;
    if (n >= 1) out[1] = 0;
    for (uint64_t i = 2; i * i <= n; ++i)
        if (out[i])
            for (uint64_t j = i * i; j <= n; j += i) out[j] = 0;
}

// Collatz stopping time and peak for every k in [1, n].
// len[k] = number of steps to reach 1, peak[k] = largest value on the orbit.
void collatz(uint64_t n, uint32_t* len, uint64_t* peak) {
    for (uint64_t k = 1; k <= n; ++k) {
        uint64_t x = k, p = k;
        uint32_t steps = 0;
        while (x != 1) {
            x = (x & 1) ? 3 * x + 1 : x >> 1;
            if (x > p) p = x;
            ++steps;
        }
        len[k] = steps;
        peak[k] = p;
    }
}

// Number of ways n = p + 2k with p prime and k >= 1, for every n in [0, limit].
// Used by the "partial primes" world. is_prime must come from sieve().
void even_plus_prime_counts(uint64_t limit, const uint8_t* is_prime, uint32_t* out) {
    std::vector<uint64_t> primes;
    for (uint64_t p = 2; p <= limit; ++p)
        if (is_prime[p]) primes.push_back(p);
    for (uint64_t n = 0; n <= limit; ++n) {
        uint32_t c = 0;
        for (uint64_t p : primes) {
            if (p + 2 > n) break;
            if (((n - p) & 1) == 0) ++c;
        }
        out[n] = c;
    }
}

// Monte Carlo estimate of the mean and variance of a linear-Gaussian
// expression sum_i (coef[i] * N(mu[i], sigma[i])) over `draws` samples.
// Deterministic for a given seed. Returns mean in out[0], variance in out[1].
void mc_linear_gaussian(uint32_t k, const double* coef, const double* mu,
                        const double* sigma, uint64_t draws, uint64_t seed,
                        double* out) {
    std::mt19937_64 rng(seed);
    std::vector<std::normal_distribution<double>> dists;
    for (uint32_t i = 0; i < k; ++i) dists.emplace_back(mu[i], sigma[i]);
    double s = 0.0, s2 = 0.0;
    for (uint64_t d = 0; d < draws; ++d) {
        double v = 0.0;
        for (uint32_t i = 0; i < k; ++i) v += coef[i] * dists[i](rng);
        s += v;
        s2 += v * v;
    }
    double mean = s / draws;
    out[0] = mean;
    out[1] = s2 / draws - mean * mean;
}

}  // extern "C"
