// 
// The normalize example from the Tapir PPOP paper converted 
// to Kokkos.  To enable kitsune+tapir compilation add the 
// flags to a standard clang compilation: 
//
//    * -fkokkos : enable specialized Kokkos recognition and 
//                 compilation (lower to Tapir).
//    * -fkokkos-no-init : disable Kokkos initialization and 
//                 finalization calls to avoid conflicts with
//                 target runtime operation. 
//    * -ftapir=rt-target : the runtime ABI to target. 
// 
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "timer.h"

using namespace std;
using namespace kitsune;

const size_t VEC_SIZE = 4096 * 20;

void random_fill(double *data, size_t N) {
  for(size_t i = 0; i < N; ++i)
    data[i] = (rand() / (double)RAND_MAX) * 10.0;
}

__attribute__((const, noinline)) double norm(const double *in, size_t N) {
  double sum = 0.0;
  for(size_t i = 0; i < N; ++i)
    sum += in[i] * in[i];
  return sqrt(sum);
}

int main (int argc, char* argv[]) {

  fprintf(stderr, "**** kitsune+tapir kokkos example: normalize (tapir paper)\n");

  double *in = new double[VEC_SIZE];
  double *out = new double[VEC_SIZE];

  random_fill(in, VEC_SIZE);
  
  double loop_secs = 0;
  for (int ii = 0; ii<4; ii++) {
      timer t;  
      {
        for (int i = 0; i<VEC_SIZE; i++) {
          out[i] = in[i] / norm(in, VEC_SIZE);
        }
      }
      loop_secs += t.seconds();
  }
  loop_secs /= 4;

  fprintf(stderr, "(%s) %lf, %lf, %lf, %lf\n", 
          argv[0], out[0], out[VEC_SIZE/4], out[VEC_SIZE/2], out[VEC_SIZE-1]); 
  fprintf(stdout, "Time: %lf\n", loop_secs);

  delete []in;
  delete []out;

  return 0;
}
