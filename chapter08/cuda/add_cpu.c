#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000

void vecAdd(float *a, float *b, int n) {
  for (int i = 0; i < n; i++) {
    b[i] = a[i] + b[i];
  }
}

int main() {
  float *a = malloc(N * sizeof(float));
  float *b = malloc(N * sizeof(float));
  // Initialize vectors
  for (int i = 0; i < N; i++) {
    a[i] = rand() / (float)RAND_MAX;
    b[i] = rand() / (float)RAND_MAX;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  vecAdd(a, b, N);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double duration =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Time taken: %f seconds\n", duration);

  free(a);
  free(b);

  return 0;
}
