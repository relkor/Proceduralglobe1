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

void initialize(double* matrix, size_t const N, int const seed)
{
    size_t k = seed;
    for (size_t y = 0; y < N; ++y) {
        for (size_t x = 0; x < N; ++x) {
            k = (k * 1021) % 256;
            matrix[y * N + x] = (float)k;
        }
    }
}


void output(double const* matrix, size_t const N)
{
    for (size_t y = 0; y < N; ++y) {
        for (size_t x = 0; x < N; ++x) {
            printf("%f ",matrix[y*N+x]);
        }
        printf("\n");
    }
   printf("\n");
   
}

//Implentation with false sharing
void matmul_naiveomp(double const* a, double const* b, double* c, size_t const N, int nthreads)
{
    int i;
#pragma omp parallel for schedule (static,1) collapse(2) num_threads(nthreads)
    for (i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            c[i * N + j] = 0;
            for (size_t k = 0; k < N; k++) {
                c[i * N + j] = c[i * N + j] + a[k + i * N] * b[j + k * N];
            }
        }
    }
}
//Implementation with Padding
void matmul_paddedomp(double const* a, double const* b, double* c, size_t const N, int nthreads)
{
    int i;
#pragma omp parallel for schedule (static,1) collapse(2) num_threads(nthreads)
    for (i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            int idx = (i* N +j)*8;//padding
            c[idx] = 0;
            for (size_t k = 0; k < N; k++) {
                c[idx] = c[idx] + a[k + i * N] * b[j + k * N];
            }
        }
    }
}

//Implementation with private variable
void matmul_privateomp(double const* a, double const* b, double* c, size_t const N, int nthreads)
{
    int i;

#pragma omp parallel for schedule (static,1) collapse(2) num_threads(nthreads)
    for (i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double temp = 0;//private variable
            for (size_t k = 0; k < N; k++) {
               temp += a[k + i * N] * b[j + k * N];
            }
            c[i * N + j] = temp;
        }
    }
}

//Implementation with increased chunk size
void matmul_bigchonkusomp(double const* a, double const* b, double* c, size_t const N, int nthreads)
{
    int i;
#pragma omp parallel for schedule (static,8) collapse(2) num_threads(nthreads)
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

    int nthreads = atoi(argv[3]);//String to int conversion
    int type = atoi(argv[2]);//String to int conversion
    size_t N = atoi(argv[argc-1]);
    double *a, *b, *c;
    size_t byteSize = N * N * sizeof(double);
    a = (double*)malloc(byteSize);
    b = (double*)malloc(byteSize);
    c = (double*)malloc(byteSize*8);//Increase size to compensate for Chunk size increase
    int seed1 = 225;
    int seed2 = 796;
    initialize(a, N, seed1);
    initialize(b, N, seed2);
    size_t r;
    double runtime = 0.0;

    if(type==0){


    for (r = 1; runtime < 100; r *= 2) {
        double start = get_time();
        for (int i = 0; i < r; ++i) {
           matmul_naiveomp(a, b, c, N,nthreads);
        }
        double end = get_time();
        runtime = end - start;
        }

    }

    if(type==1){
        for (r = 1; runtime < 100.0; r *= 2) {
             double start = get_time();
             for (int i = 0; i < r; ++i) {
               matmul_paddedomp(a, b, c, N,nthreads);
        }
        double end = get_time();
        runtime = end - start;
        }
    }

    if(type==2){


    for (r = 1; runtime < 100.0; r *= 2) {
        double start = get_time();
        for (int i = 0; i < r; ++i) {
           matmul_privateomp(a, b, c, N,nthreads);
        }
        double end = get_time();
        runtime = end - start;
        }
    }

    if(type==3){

    for (r = 1; runtime < 100.0; r *= 2) {
        double start = get_time();
        for (int i = 0; i < r; ++i) {
           matmul_bigchonkusomp(a, b, c, N,nthreads);
        }
        double end = get_time();
        runtime = end - start;
        }
    }
     r = r / 2;
    runtime = runtime / (double)r;
    printf("%f", runtime);


    //uncomment the following for output
    /*output(a, N);
    output(b, N);
    output(c, N);*/
/*
    free(a);
    free(b);
    free(c);
    return 0;
    */
}
