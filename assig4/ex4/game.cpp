#include <cstdlib>
#include <cstdio>

#include "Stopwatch.h"

// This block enables to compile the code with and without the likwid header in place
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

class RigidBody {
public:
  double x,y,z;

  RigidBody() {
    x = drand48();
    y = drand48();
    z = drand48();
  }

  inline void move(double dx, double dy, double dz) {
    x += dx;
    y += dy;
    z += dz;
  }
};

void gravity(double dt, RigidBody** bodies, int N) {
  LIKWID_MARKER_START("gravity");
  for (int n = 0; n < N; ++n) {
    bodies[n]->move(0.0, 0.0, 0.5 * 9.81 * dt * dt);
  }
  LIKWID_MARKER_STOP("gravity");
}

int main() {
  int N = 10000;
  int T = 10;

  RigidBody** bodies = new RigidBody*[N];
  for (int i = 0; i < N; ++i) {
    bodies[i] = new RigidBody[i];
  }
  
  Stopwatch sw;
  sw.start();

  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
  
  for (int t = 0; t < T; ++t) {
    gravity(0.001, bodies, N);
  }

  LIKWID_MARKER_CLOSE;
  
  double time = sw.stop();
  printf("Time: %lf s, BW: %lf\n MB/s", time, T*N*sizeof(double)*1e-6 / time);

  for (int i = 0; i < N; ++i) {
    delete bodies[i];
  }
  delete[] bodies;

  return 0;
}
