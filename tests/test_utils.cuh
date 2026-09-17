#pragma once
#include <cstdio>
#include <cmath>
#include <cstddef>

static inline void fill(float* p, size_t n, unsigned& seed){
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;    // LCG: reproducible, no <random>
        p[i] = ((float)(seed >> 8) / 8388608.0f) - 1.0f;   // ~[-1,1]
    }
}

static inline void fill_labels(int* y, size_t n, int C, unsigned& seed){
    for(size_t i = 0; i < n; ++i){
        seed = seed * 1664525u + 1013904223u;
        y[i] = (int)((seed >> 8) % (unsigned)C);
    }
}

static inline bool close(const float* got, const float* ref, size_t n, double atol = 1e-5, double rtol = 1e-4){
    double worst = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i){
        double err = fabs((double)got[i] - (double)ref[i]);
        double tol = atol + rtol * fabs((double)ref[i]);
        double score = err / tol;
        if (score > worst) { worst = score; bad = i; }
    }
    if (worst > 1.0){
        printf("mismatch at [%zu]: got %.9g want %.9g (abs %.3g, rel %.3g)\n",
            bad, got[bad], ref[bad],
            fabs((double)got[bad] - ref[bad]),
            fabs((double)got[bad] - (double)ref[bad]) / (fabs((double)ref[bad]) + 1e-30));
        return false;
    }
    return true;
}