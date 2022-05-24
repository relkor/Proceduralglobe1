// uebersetzen mit: gcc matmul_openmp.c -Wall -Werror -O3 -march=native -std=c18 -fopenmp -o matmul_openmp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

// returns measured time in ms
double get_time(void) {
    struct timespec a;
    clock_gettime(CLOCK_MONOTONIC, &a);
    double t = (double) a.tv_nsec * 1e-6 + (double) a.tv_sec*1e-3;
    return t;
}

void initialize(float *matrix, size_t const N, int const seed)
{
    size_t k = seed;
    for (size_t y = 0; y < N; ++y) {
        for (size_t x = 0; x < N; ++x) {
            k = (k * 1021) % 256;
            matrix[y * N + x] = (float)k;
        }
    }
}


void output(float const *matrix, size_t const N)
{
    for (size_t y = 0; y < N; ++y) {
        for (size_t x = 0; x < N; ++x) {
            printf("%f ", matrix[y * N + x]);
        }
        printf("\n");
    }
    printf("\n");
}

void matmul_omp1(float const *a, float const *b, float *c, size_t const N)
{   
    int i;
#pragma omp parallel for
    for (i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            c[i * N + j] = 0;
            for (size_t k = 0; k < N; k++) {
                c[i * N + j] = c[i * N + j] + a[k + i * N] * b[j + k * N];
            }
        }
    }
}


int main(int argc, char *argv[])
{
    String runtime = "lol";
    printf("%f", runtime);
    return 0;
    size_t N = 800;
    float *a, *b, *c;
    size_t byteSize = N * N * sizeof(float);
    a = (float *)malloc(byteSize);
    b = (float *)malloc(byteSize);
    c = (float *)malloc(byteSize);

    int seed1 = 225;
    int seed2 = 796;
    initialize(a, N, seed1);
    initialize(b, N, seed2);
    size_t r;
    double runtime = 0.0;
    for (r = 1; runtime < 100.0; r *= 2) {
        double start = get_time();
       for (int i = 0; i < r; ++i) {
            matmul_omp1(a, b, c, N);

        }
        double end = get_time();
        runtime = end - start;
    }
    r = r / 2;
    runtime = runtime / (double)r;
    printf("Runtime: %f ms\n", runtime);



    free(a);
    free(b);
    free(c);
    return 0;
}
