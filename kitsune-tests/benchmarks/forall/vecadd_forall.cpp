// 
// Simple example of an element-wise vector sum.  
// To enable kitsune+tapir compilation add the flags to a standard 
// clang compilation: 
//
//    * -fkokkos : enable specialized Kokkos recognition and 
//                 compilation (lower to Tapir).
//    * -fkokkos-no-init : disable Kokkos initialization and 
//                 finalization calls to avoid conflicts with
//                 target runtime operation. 
//    * -ftapir=rt-target : the runtime ABI to target. 
// 
#include <cstdio>
#include <cstdlib>
#include <kitsune.h>

#include "timer.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 1024 * 1024 * 256;

void random_fill(float *data, size_t N) {
  for(size_t i = 0; i < N; ++i) 
    data[i] = rand() / (float)RAND_MAX;
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "kitsune+tapir kokkos example: element-wise vector addition\n");
  
  float *A = new float[VEC_SIZE];
  float *B = new float[VEC_SIZE];
  float *C = new float[VEC_SIZE];

  random_fill(A, VEC_SIZE);
  random_fill(B, VEC_SIZE);
  
  double loop_secs = 0;
  for (int ii = 0; ii<4; ii++) {
      timer t;  
      {
        forall (int i = 0; i<VEC_SIZE; i++) {
          C[i] = A[i] + B[i];
        }
      }
      loop_secs += t.seconds();
  }
  loop_secs /= 4;

  // Note: If we don't use the outputs there are cases where tapir+kitsune 
  // will simply remove the entire parallel loop above... 
  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
          argv[0], C[0], C[VEC_SIZE/4], C[VEC_SIZE/2], C[VEC_SIZE-1]);   
  
  fprintf(stdout, "Time: %lf\n", loop_secs);

  delete []A;
  delete []B;
  delete []C;

  return 0;
}
