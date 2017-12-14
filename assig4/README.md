# Assignment 4


## Part 1: Superuseful application - GNU gprof

## Part 2: Quicksort - Intel VTune Amplifier XE

## Part 3: CG - Scalasca

## Part 4: Likwid-perfctr

### 1. The file game.cpp could be part of a physics engine for a computer game. As in classic object oriented programming styles, it creates a heap-allocated object for every rigid body. Here, we want to analyse its performance characteristics using LIKWID.

#### a. Using the marker API of LIKWID allows the measuring of specific code regions. Introduce a marker for the gravity function. What do you need to do in order for the markers to work (compilation options, command line arguments)?

To introduce a marker the cpp file needs to be modified.
*likwid.h* needs to be included and the marker needs to be added.

```c
  ...

  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
  
  for (int t = 0; t < T; ++t) {
    gravity(0.001, bodies, N);
  }

  LIKWID_MARKER_CLOSE;
  
  ...
```

```c
void gravity(double dt, RigidBody** bodies, int N) {
  LIKWID_MARKER_START("gravity");
  for (int n = 0; n < N; ++n) {
    bodies[n]->move(0.0, 0.0, 0.5 * 9.81 * dt * dt);
  }
  LIKWID_MARKER_STOP("gravity");
}
```

The code has to be linked against the LIKWID library and Pthreads needs to be enabled during linking.
The modified Makefile looks like this:

```
LIKWID_INCLUDE=/lrz/sys/tools/likwid/likwid-4.1/bin/../include/
LIKWID_LIB=/lrz/sys/tools/likwid/likwid-4.1/bin/../lib/

LIKWID_FLAGS = -DLIKWID_PERFMON -L$(LIKWID_LIB) -I$(LIKWID_INCLUDE) -llikwid

CXX=icpc
CXXFLAGS=-O3 -xHost -pthread $(LIKWID_FLAGS)
LDFLAGS=-lrt  $(LIKWID_FLAGS)
...
```

The command to run likwid-perfctr for a serial application with the marker API enabled in an interactive shell on the cluster looks like this:

*srun likwid-perfctr -C S0:0 -g <groupname> -o out.txt -m <EXEC>*

#### b.  Use *likwid-perfctr -a* to get a list of all available event groups. Which event groups are relevant for this kind of application and which are not? Give a brief explanation for every event group.

| Group Name  	| Description                                                                                                                                                                             	|
|-------------	|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| BRANCH      	| Branch prediction miss rate/ratio; How often a branch or a mispredicted branch occured per instruction retired in total; relevant                                                       	|
| CACHES      	| Cache bandwidth in MBytes/s; Measures cache transfers between L1 and Memory; relevant                                                                                                   	|
| CBOX        	| CBOX related data and metrics, CBOXes mediate traffic from L2 cache to the segmented L3 cache; Each CBOX is responsible for one segment (2.5 MByte); not relevant (serial application)  	|
| CLOCK       	| Power and Energy consumption; Monitors the consumed energy on the package level with RAPL interface; relevant if going for energy-awareness (but also implemented in ENERGY)            	|
| DATA        	| Load to store ratio; not really relevant                                                                                                                                                	|
| ENERGY      	| Power and Energy consumption; Monitors the consumed energy on the package level and DRAM LEVEL with RAPL interface; relevant if going for energy-awareness                              	|
| FALSE_SHARE 	| Measures L3 traffic induced by false-sharing; relevant                                                                                                                                  	|
| FLOPS_AVX   	| Packed AVX MFLOP/s; Approximate counts of AVX & AVX2 256-bit instructions; relevant                                                                                                     	|
| HA          	| Main memory bandwidth in MBytes/s seen from Home agent (central unit that is responsible for the protocol side of memory interactions); not really relevant (not many memory transfers) 	|
| ICACHE      	| L1 instruction cache metrics; relevant                                                                                                                                                  	|
| L2          	| L2 cache bandwidth in MBytes/s; relevant                                                                                                                                                	|
| L2CACHE     	| L2 cache miss rate/ratio; relevant                                                                                                                                                      	|
| L3          	| L3 cache bandwidth in MBytes/s; relevant                                                                                                                                                	|
| L3CACHE     	| L3 cache miss rate/ratio; relevant                                                                                                                                                      	|
| MEM         	| Main memory bandwidth in MBytes/s; Same metrics as HA group; not really relevant (not many memory transfers)                                                                            	|
| NUMA        	| Local and remote memory accesses; not relevant (serial application)                                                                                                                     	|
| QPI         	| QPI Link Layer data; not relevant (serial application)                                                                                                                                  	|
| RECOVERY    	| Recovery duration after SSE exceptions, memory disambiguations, etc; not relevant                                                                                                       	|
| SBOX        	| Ring Transfer bandwidth between the socket local ring(s); not relevant (serial application)                                                                                             	|
| TLB_DATA    	| L2 data TLB miss rate/ratio; relevant                                                                                                                                                   	|
| TLB_INSTR   	| L1 Instruction TLB miss rate/ratio; relevant                                                                                                                                            	|
| UOPS        	| UOPs execution info; Measures issued, executed and retired uOPS; relevant                                                                                                               	|
| UOPS_EXEC   	| Ratios of used and unused cycles regarding the execution stage in the pipeline; relevant                                                                                                	|
| UOPS_ISSUE  	| Like EXEC, but issue stage; relevant                                                                                                                                                    	|
| UOPS_RETIRE 	| Retire stage; relevant                                                                                                                                                                  	|

#### c. Run the application and measure relevant event groups. What would be your
suggestions for improving the performance of the application?

TODO (results already in git)
use AVX instructions? l2 misses? reduce unused uops cycles?

### 2. The file dtrmv.cpp contains a routine for upper-triangular matrix times vector multiplication. We want to investigate possible load imbalances, as the routine does not scale well to multiple cores.

