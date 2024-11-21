#include <math.h>
#include <stdio.h>
#include <time.h>

#define N 100000000

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void) {
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = rand() / (float)RAND_MAX;
    y[i] = rand() / (float)RAND_MAX;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &end);
  double duration =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Time taken: %f seconds\n", duration);

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
