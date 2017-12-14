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

#### a. Calculate the number of flops for every core manually and explain the load imbalance

Flops for upper-triangular matrix * vector are $$O(2*(\sum_{i=0}^{N} N-i))$$

One *Intel Xeon E5-2697 v3* has 14 cores. 

Total number of flops: 100010000 flops  

Calculating the Number of flops for 14 threads in dtrmv function:

The outer loop is divided equally:

*#pragma omp parallel for*  
*for (int i = 0; i < N; ++i)*

10000/14 = 714 R:4

4 cores execute 715 loop runs, 10 execute 714 loop runs.

Core 0 executes loop runs i=0 to i=714  
Core 1: 715 to 1429  
Core 2: 1430 to 2144  
Core 3: 2145 to 2859  
Core 4: 2860 to 3573  
Core 5: 3574 to 4287  
Core 6: 4288 to 5001  
Core 7: 5002 to 5715  
Core 8: 5716 to 6429  
Core 9: 6420 to 7143  
Core 10: 7144 to 7857  
Core 11: 7858 to 8571  
Core 12: 8572 to 9285  
Core 13: 9286 to 9999  

Then we can calculate the number of flops per core using $$O(2*(\sum_{i=start}^{end} N-i))$$

For Core 0:
$$2*(\sum_{i=0}^{714} 10000-i)= 2 * 6894745 = 13789490 flops$$

Core 0: 13789490 flops  
Core 1: 12767040 flops  
Core 2: 11744590 flops  
Core 3: 10722140 flops  
Core 4: 9686838 flops  
Core 5: 8667246 flops  
Core 6: 7647654 flops  
Core 7: 6628062 flops  
Core 8: 5608470 flops  
Core 9: 4660388 flops  
Core 10: 3569286 flops  
Core 11: 2549694 flops  
Core 12: 1530102 flops  
Core 13: 510510 flops  

Because the matrix is upper-triangular every chunk has a different number of flops using the default static scheduler. So every core has to execute less flops then the previous one even though the chunk size is the same.
This means that core 13 is most likely finished a long time before core 0 is.
This is the load imbalance.

#### b. Instead of hand calculation, it would be convenient to use hardware counters for the number of flops. However, these are known to be inexact since Sandy Bridge. Insert a marker for the dtrmv routine and measure the number of flops using LIKWID. Repeat the measurement for different numbers of threads. Are the measurements
useful to find the load imbalance?

Haswell has no FLOP events, only AVX_INSTS  
Using FLOPS_AVX

cpp file is prepared for execution and should compile

#### c. Is *INSTR RETIRED ANY* a useful alternative?

#### d. Modify the program such that the load is approximately balanced.

dynamic scheduler
