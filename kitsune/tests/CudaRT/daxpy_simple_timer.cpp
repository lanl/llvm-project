#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <chrono>

#include <cilk/cilk.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

// Initialize the input data.
void initialize(float *x, int n) {
  // Initialize data[] such that data[i] = i / 10.0.  This
  // array allows us to check the result of the sum easily and observe
  // errors that arise in the floating-point calculation.
  for (int i = 0; i < n; ++i) {
    x[i] = ((float)i) / 10.0;
  }
}

__global__
void daxpy(float* y, float* x, int n, float a) {

  initialize(x, n);

  float sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += y[i];
  printf("Before: sum = %f\n", sum);

  auto start = std::chrono::steady_clock::now();
  #pragma cilk grainsize 2
  cilk_for (int i = 0; i < n; ++i) {
    y[i] += a * x[i];
  }

  auto stop = std::chrono::steady_clock::now();
  
  sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += y[i];
  printf("After: sum = %f\n", sum);

  std::chrono::duration<double> elapsed_seconds = stop-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

int main(int argc, char *argv[]) {
  int n = 2048;
  float a = 1.5;
  if (argc > 1)
    n = atoi(argv[1]);
  if (argc > 2)
    a = atof(argv[2]);

  printf("n = %d\n", n);

  float *y, *x;

  daxpy(y, x, n, a);

  return 0;
}

