//
// Kokkos implementation of work inspired by:
//
//   - https://fabiensanglard.net/revisiting_the_pathtracer/
//
//
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <kitsune/timer.h>
#include "Kokkos_DualView.hpp"
#include <iostream>
#include <fstream>

#define WIDTH  1920
#define HEIGHT 1080
#define BPP 3
typedef Kokkos::DualView<unsigned char**[3], Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> DualViewVector;

struct Vec {
 float x,y,z;
 KOKKOS_INLINE_FUNCTION Vec(float v = 0) {x = y = z = v;}
 KOKKOS_INLINE_FUNCTION Vec(float a, float b, float c = 0) {x = a;y = b;z = c;}
 KOKKOS_INLINE_FUNCTION Vec operator+(const Vec r) const  { return Vec(x + r.x , y + r.y , z + r.z); }
 KOKKOS_INLINE_FUNCTION Vec operator*(const Vec r) const { return   Vec(x * r.x , y * r.y , z * r.z); }
 KOKKOS_INLINE_FUNCTION float operator%(const Vec r) const {return     x * r.x + y * r.y + z * r.z;}
 KOKKOS_INLINE_FUNCTION Vec operator!() { return *this * (1.0/sqrtf(*this % *this)); }
};

KOKKOS_INLINE_FUNCTION float randomVal(unsigned int& x) {
   x = (214013*x+2531011);
   return ((x>>16)&0x7FFF) / (float)66635;
}

// Rectangle CSG equation. Returns minimum signed distance from
// space carved by
// lowerLeft vertex and opposite rectangle vertex upperRight.
KOKKOS_INLINE_FUNCTION float BoxTest(const Vec& position, Vec lowerLeft, Vec upperRight) {
 lowerLeft = position + lowerLeft * -1.0f;
 upperRight = upperRight + position * -1.0f;
 return -fminf(
     fminf(fminf(lowerLeft.x, upperRight.x), fminf(lowerLeft.y, upperRight.y)),
     fminf(lowerLeft.z, upperRight.z));
}

#define HIT_NONE 0
#define HIT_LETTER 1
#define HIT_WALL 2
#define HIT_SUN 3

// Sample the world using Signed Distance Fields.
KOKKOS_INLINE_FUNCTION float QueryDatabase(const Vec& position, int &hitType) {
 float distance = 1e9;//FLT_MAX;
 Vec f = position; // Flattened position (z=0)
 f.z = 0;

 static const float lines[10*4] = {
                -21, 0, -21, 16,
                -21, 0, -15, 0,
                -11, 0, -7, 16,
                -3, 0, -7, 16,
                -6.5, 5, -9.5, 5,
                0, 0, 0, 16,
                6, 0, 6, 16,
                0, 16, 6, 0,
                9, 0, 9, 16,
                9, 0, 15, 0
        };
 for (int i = 0; i < sizeof(lines); i += 4) {
   Vec begin = Vec(lines[i], lines[i + 1]) * .5;
   Vec e = Vec(lines[i + 2], lines[i + 3]) * .5 + begin * -1;
   Vec o = f + (begin + e * fminf(-fminf((((begin + f * -1) % e )/(e % e)), 0),1)) * -1;
   distance = fminf(distance, o % o); // compare squared distance.
 }
 distance = sqrtf(distance); // Get real distance, not square distance.
 distance = powf(powf(distance, 8)+powf(position.z, 8), 0.125f) - 0.5f;
 hitType = HIT_LETTER;

 float roomDist ;
 roomDist = fminf(-fminf(
         BoxTest(position, Vec(-30, -.5, -30), Vec(30, 18, 30)),
         BoxTest(position, Vec(-25, 17, -25), Vec(25, 20, 25))
     ),
     BoxTest( // Ceiling "planks" spaced 8 units apart.
         Vec(fmodf(fabsf(position.x), 8), position.y, position.z),
         Vec(1.5, 18.5, -25),
         Vec(6.5, 20, 25)
     )
 );
 if (roomDist < distance){distance = roomDist, hitType = HIT_WALL;}
 float sun = 19.9 - position.y; // Everything above 19.9 is light source.
 if (sun < distance){distance = sun, hitType = HIT_SUN;}
 return distance;
}

// Perform signed sphere marching
// Returns hitType 0, 1, 2, or 3 and update hit position/normal
KOKKOS_INLINE_FUNCTION int RayMarching(const Vec& origin, const Vec& direction, Vec& hitPos, Vec& hitNorm) {
 int hitType = HIT_NONE;
 int noHitCount = 0;

 // Signed distance marching
 float d; // distance from closest object in world.
 for (float total_d = 0; total_d < 100; total_d += d) {
   if ((d = QueryDatabase(hitPos = origin + direction * total_d, hitType)) < .01 || ++noHitCount > 99)
     return hitNorm =
                !Vec(QueryDatabase(hitPos + Vec(.01, 0), noHitCount) - d,
                     QueryDatabase(hitPos + Vec(0, .01), noHitCount) - d,
                     QueryDatabase(hitPos + Vec(0, 0, .01), noHitCount) - d),
            hitType; // Weird return statement where a variable is also
                     // updated.
 }
 return 0;
}

KOKKOS_INLINE_FUNCTION Vec Trace(Vec origin, Vec direction, unsigned int& rn) {
  Vec sampledPosition;
  Vec normal;
  Vec color = 0;
  Vec attenuation = 1;
  Vec lightDirection(!Vec(0.6f, 0.6f, 1.0f)); // Directional light

  for (int bounceCount = 8; bounceCount--;) {
    int hitType = RayMarching(origin, direction, sampledPosition, normal);
    if (hitType == HIT_NONE)
      break;                     // No hit. This is over, return color.
    if (hitType == HIT_LETTER) { // Specular bounce on a letter. No color acc.
      direction = direction + normal * (normal % direction * -2);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f; // Attenuation via distance traveled.
    }
    if (hitType == HIT_WALL) { // Wall hit uses color yellow?
      float incidence = normal % lightDirection;
      float p = 6.283185f * randomVal(rn);
      float c = randomVal(rn);
      float s = sqrtf(1 - c);
      float g = normal.z < 0 ? -1 : 1;
      float u = (-1 / (g + normal.z));
      float v = normal.x * normal.y * u;
      float cosp;
      float sinp;
      sincosf(p, &sinp, &cosp);
      direction =
          Vec(v, g + normal.y * normal.y * u, -normal.y) * (cosp * s) +
          Vec(1 + g * normal.x * normal.x * u, g * v, -g * normal.x) *
          (sinp * s) +
          normal * sqrtf(c);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f;
      if (incidence > 0.0f &&
          RayMarching(sampledPosition + normal * 0.1f, lightDirection,
                      sampledPosition, normal) == HIT_SUN)
        color = color + attenuation * Vec(500, 400, 100) * incidence;
    }
    if (hitType == HIT_SUN) { //
      color = color + attenuation * Vec(50, 80, 100);
      break; // Sun Color
    }
  }
  return color;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  kitsune::timer t;
  {
    DualViewVector img = DualViewVector("img", WIDTH, HEIGHT);
    int samplesCount = 1 << 7;
    if (argc > 1 ) {samplesCount = atoi(argv[1]);}
    //img.modify_device();
    Kokkos::parallel_for(WIDTH*HEIGHT, KOKKOS_LAMBDA(const int i) {
        const int x = i % WIDTH;
        const int y = i / WIDTH;
        unsigned int v = i;
        //curand_init( x * y, 0, 0 , &states[y]);
        const Vec position(-12.0f, 5.0f, 25.0f);
        const Vec goal = !(Vec(-3.0f, 4.0f, 0.0f) + position * -1.0f);
        const Vec left = !Vec(goal.z, 0, -goal.x) * (1.0f / WIDTH);
        // Cross-product to get the up vector
        const Vec up(goal.y *left.z - goal.z * left.y, goal.z *left.x -
                     goal.x * left.z, goal.x *left.y - goal.y * left.x);
        Vec color;
        for (int p = samplesCount; p--;) {
          Vec rand_left = Vec(randomVal(v), randomVal(v), randomVal(v))*.001;
          color = color + Trace(position, !((goal+rand_left) + left *
                                            ((x+randomVal(v)) - WIDTH / 2.0f + randomVal(v)) + up *
                                            ((y+randomVal(v)) - HEIGHT / 2.0f + randomVal(v))), v);
        }
        // Reinhard tone mapping
        color = color * (1.0f / samplesCount) + 14.0f / 241.0f;
        Vec o = color + 1.0f;
        color = Vec(color.x / o.x, color.y / o.y, color.z / o.z) * 255.0f;
        img.d_view(x,y,0) = color.x;
        img.d_view(x,y,1) = color.y;
        img.d_view(x,y,2) = color.z;
      });
    double loop_secs = t.seconds();
    std::cout << "running time: " << loop_secs << std::endl;

    std::ofstream myfile;
    myfile.open ("example.ppm");
    //img.sync_host();
    myfile << "P6 " << WIDTH << " " << HEIGHT << " 255 ";
    for (int y = HEIGHT; y--;) {
      for (int x = WIDTH; x--;) {
        //int offset = y * w * BPP + x * BPP;
        myfile << (char)img.h_view(x,y,0) << (char)img.h_view(x,y,1) << (char)img.h_view(x,y,2);
      }
    }
  }
  Kokkos::finalize();
  return EXIT_SUCCESS;
}
