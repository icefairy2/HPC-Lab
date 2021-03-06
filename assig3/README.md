# Assignment 3


## Part 1:
Amdahl's Law:
![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7B1%7D%7Bs%20%2B%5Cfrac%7B1%20-%20s%7D%7Bp%7D%20%7D%20%7D%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

Parallel efficiency:  
Eff = Sp/p
#### 1. Determine the maximum number of processes with s=10% and Eff>=70%.
Given that the efficiency should be *at least* 70%, we start from the equation Eff >= 0.7, which is equivalent to: Sp/p >= 0.7

Substituting the given values our equation has the following form:

![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Cfrac%7B%5Cfrac%7B1%7D%7B0.1%20%2B%5Cfrac%7B1%20-%200.1%7D%7Bp%7D%20%7D%20%7D%7Bp%7D%20%20%5Cgeq%200.7&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

Simplifying it we get to the relation:

        p <= 37/7, with 37/7=~5.28

Having p <= 5.28, we conclude that the maximum number of processes p to achieve a parallel efficiency of at least 70%, having a sequential portion of 10% is:

        pMax = 5
 
#### 2. Explain why Amdahl's law is pessimistic. Are there other laws in order to classify the parallel behaviour of an application?
In the previous section we saw that to achieve a parallel efficiency of at least 70% the maximum number of processes that could be used is 5. This is in fact a small number and larger numbers of processes would lead to less efficiency.
We see that for large numbers of processes the speedup would only depend on the serial section of the application, having the form Sp = 1/s, thus the speedup becomes constant from a given p.
Amdahl's law is pessimistic regarding massive parallel computation because the number of processes p has smaller weighing factor in determining the speedup than the sequential part of the code.
The ideal linear speedup would be achieved if s = 0, in which case Sp = p, in other words Sp/p = 1 resulting in a 100% efficiency.
When we increase the value of s at around 20% we already get Sp < 5 independently from p. Here, the efficiency decreases with increasing value of p.

Amdahl's law assumes the problem size to remain constant, but in most cases bigger computers are used to solve bigger problems, not to solve old problems faster. Small fractions of sequential code are often inevitable (i.e. reading input data). 
In these cases by Amdahl's law if the sequential part of the program does not increase when increasing the input (or increases sublinearly) applications can run on a large number of cores, even if the speedup remains constant from a given number of processes.

Another law to classify the parallel behaviour of an application is Gustafson's law. Its formula looks like the following:

        Sp = p + (1 - p) * s
        
While Amdahl's law gives the theoretical speedup in latency of the execution of a task at fixed workload, Gustaffson's law fixes the execution time.

## Part 2: Probing the network

#### 1. Do a literature research on the Omni-Path in general and the Omni-Path network used in the Linux Cluster. 

Omni-Path is a high-performance communication architecture by Intel. It competes with InfiniBand and is designed specifically for HPC clusters. Omni-Path and InfiniBand both run at 100Gbps but Omni-Path implements a few optimizations for HPC applications. These include changing the Forward Error Correction in the Link Transfer Layer for Link Transfer Packets to a simple 14-bit CRC, which reduces latencies. It also includes Traffic Flow Optimization, which allows sending multiple Messages in a single packet and Quality of Service changes, that allow the interruption of transmissions in order to send high priority messages. New routing options like Adaptive Routing and Dispersive Routing were also introduced.

The Linux Cluster consists of 148 nodes with *Intel Xeon Phi 7210-F* processors, that have intergrated 2 port Omni-Path Fabrics. The nodes are connected by 10 + 4 48 port *100SWE48* switches, that support data rates of 100Gbps per port. The Omnipath network has a Fat-Tree-Topology. The compute nodes are connected within the Ompi-Path fabric via the integrated 2 ports and the entire Omni-path fabric is connected with a blocking factor of 2:1. 

#### 1. Explain why the bandwidth depends on the message size

*Data transfer time = latency + message size / bandwith*

For short messages the latency dominates the transfer time and for long messages the bandwith term dominates. For MPI applications a larger message size often yields better performance due to the high available bandwith. Omni-Path supports MTU sizes from *2048B* up to *8192B*, so it benefits from larger message sizes.

#### 1. What latencies and peak bandwidths can you expect?

The peak bandwidth of each port is 100 Gbps and the bidirectional bandwith of a node to the Interconnect is 25 Gbps.
The bisection bandwidth of the Interconnect is 1.6 TB/s.
The latency oft he Interconnect is 2.3µs.

#### 2. Draw a picture of the topology

![alt text](https://i.imgur.com/uQipkHc.png "Cluster topology")

#### 3. Compute the bisection width and bisection bandwidth of 16 nodes and of 32 nodes w.r.t. the topology. Assume that 16 nodes are connected to each leaf switch.

To separate the network into two equal parts, we would have to cut a minimum amount of 20 edges which are between 2-2 spine switches being connected to 5 leaf switches on one side, and cutting the connection to the other half.
This way, the bisection width would be equal to 20.
The bisection bandwidth is the sum of the single badwidth of all edges along the cut, and because the edges were 4 port links, we would have 20 * 4 * (bandwidth).
In our case the bandwith is 100Gbit so the final value of the bisection bandwidth is 8000 Gbit.

#### 4. Give a short explanation of the sub-benchmakrs in the *Intel MPI Benchmarks*

There are benchmarks for MPI-1, MPI-2 and MPI-3 functions.
The MPI-1 benchmarks will be explained in more detail.
MPI-2 benchmarks can be devided into IMB-EXT and IMB-IO benchmarks.
MPI-3 benchmarks can be devided into IMB-NBC and IMB-RMA benchmarks.


##### MPI-1 Benchmarks: 

There are three classes of benchmarks:
Single Transfer, Paralell Transfer and Collective benchmarks

###### Single Transfer Benchmarks:
Single transfer benchmarks involve 2 active processes into communication and other processes wait for the completion of the communication. The timing is averaged between 2 processes. Benchmarks are run with varying message lengths.

The throughput is measured in MBps.

*throughput = X/time*

*X* is message length in byte

*time* is measured in μsec

e.g. *PingPong*: Measures startup and throughput of a single message sent between 2 processes

###### Parallel Transfer Benchmarks:
Parallel transfer benchmarks involve more than 2 active processes into communication. Benchmarks are run with varying message lengths. The timing is averaged over multiple samples.
*MPI_BYTE* is the basic MPI data type used for all messages. For the throughput calculations the multiplicity of messages outgoing from or incoming to a process is taken into account.

The throughput is measured in MBps.

*throughput = nmsg*X/time*

*nmsg* is multiplicity of messages

e.g. *Sendrecv*: Processes form a periodic communication chain where each process sends a message to the right neighbor and receives a message from the left neighbor in the chain.

###### Collective Benchmarks:
Collective benchmarks measure MPI collective operations. Benchmarks are run with varying message lengths. The timing is averaged over multiple samples.
For pure data movement functions the basic MPI data type is MPI_BYTE and for reductions it is MPI_FLOAT.
Collective benchmarks show bare timings.

e.g. *Bcast*: The root process broadcasts a message to all other processes.

#### 5. Run the *Intel MPI Benchmarks* on the Linux Cluster using different number of nodes.

A run of the benchmark program executes the benchmarks for different message lengths. For lenghts up to 32768 bytes there are 1000 repetitions. Then the number of repetitions are going down from 640 for 65536 bytes to 10 repetitions for 4194304 bytes. Because there seem to be fluctuations in execution time for large message sizes on the cluster, the benchmarks are run 5 times for the tested numbers of nodes and the results are then averaged. E.g. PingPong fluctuated between *2009.73 Mbytes/sec* and *10232.48 Mbytes/sec* for the largest message size in the five benchmark runs.

By default the benchmark is run for a different number of processes, e.g. 2,4,8,etc.
It doesn't seem to make a big difference if e.g. a benchmark is run with 64 allocated nodes and 16 processes or with 16 allocated nodes and 16 processes, so only the results with *number of nodes = number of processes* are included in the graphs.

The output of the runs can be found in the *ex2/output* folder and the data used for creating the graphs can be found in the *mpi-benchmarks.xlsx* in the *ex2* folder.

###### Single Tranfer Benchmarks

The Single Transfer Benchmarks run by the MPI1 benchmark tests are PingPong and PingPing.
Because Single Transfer Benchmarks only involve 2 active processes, only the results for 2 nodes are included in the graphs.

![alt text](https://i.imgur.com/1kJX72Y.png "PingPong")

![alt text](https://i.imgur.com/38AJc0i.png "PingPing")

For PingPong and PingPing the bandwidth grows with larger message sizes. For the largest message size of 4194304 bytes the average bandwidth becomes slower, because some of the runs slowed down to 2009.73 Mbytes/sec (PingPong) and 1071.34 Mbytes/sec (PingPing). Nonetheless the largest bandwidths are also achieved with the largest message size with 10232.48 Mbytes/sec and 10041.20 Mbytes/sec respectively.

###### Parallel Transfer Benchmarks

The executed Parallel Transfer Benchmarks are Sendrecv and Exchange. They are executed for 1 up to 128 nodes. Due to cluster usage there are only 2 executions for 128 nodes, as getting 128 nodes on the cluster isn't easy. But because the benchmarks use multiple repetitions, the results should still be valid.

![alt text](https://i.imgur.com/DBFvCG0.png "Sendrecv")

For Sendrecv the achieved bandwidth for 4 to 32 nodes is relatively similar and lowers a bit for the largest message size.
For 2 nodes the bandwidth increases a bit for the largest message length, probably due to the fact that Sendrecv is equivalent to PingPing for 2 processes.
The average bandwidth of 64 and 128 nodes drops drastically for the 2 largest message sizes.
For 2097152 bytes the maximum bandwidth for 64 nodes is 1546.77 Mbytes/sec and for 128 nodes it was only 1014.6 Mbytes/sec.
For 4194304 bytes the maximum bandwidths were 4716.66 Mbytes/sec and 1435.99 Mbytes/sec respectively, which is significantly slower than the benchmarks with fewer nodes.


![alt text](https://i.imgur.com/VUEhW6L.png "Exchange")

The results of the Exchange benchmark are relatively similar to the Sendrecv benchmark.
Here, the performance for 64 nodes only drops for the largest messages.
For 2097152 bytes the maximum bandwidth for 64 nodes is 12004.22 Mbytes/sec, which is about the same maximum bandwidth as for runs with fewer nodes.
For 128 nodes it was only 1713.34 Mbytes/sec.
For 4194304 bytes the maximum bandwidths were 4305.37 Mbytes/sec and 1423.89 Mbytes/sec respectively, which is significantly slower than the benchmarks with fewer nodes.


###### Collective Benchmarks

The MPI1 benchmark program executes 13 collective benchmarks.
Allreduce, Allgather and Bcast are used for the graphs.

![alt text](https://i.imgur.com/LZ36mH4.png "Allreduce")

The average execution time for Allreduce grows for larger message sizes and larger amounts of nodes, which is expected.
The growing rate becomes larger for larger message lenghts.

![alt text](https://i.imgur.com/rQhZUuX.png "Allgather")

For Allgather it looks similar, but with less change in the growth rates.
Also the differences in execution times between different amount of nodes are significantly larger than for the Allreduce benchmark.

![alt text](https://i.imgur.com/LIgcYgq.png "Bcast")

For Bcast the growth rates for the different number of nodes are pretty similar.
Except for the execution time for 32 nodes and 2097152 bytes, which seems to be spiking a bit. That could probably be neutralized using an even larger sample size.
Also, for 8 nodes the time spikes for the two largest message sizes, which is due to a benchmark run averagin with 5522.02 µsecs and 8859.15 µsecs respectively.

## Part 3: Broadcast

### How to run

- copy the `broadcast` directory in the cluster.
- run `make` to compile the program
- adjust `script.sh` `-o` and `-D` parameters to point to your account directory
- run `sbatch script.sh`

### Implementation details

We want to measure the bandwith of the system by broadcasting an array held initially
by the root process to all the other processes.There are four broadcast
implementations to be compared:

#### Naive

For the naive approach, the root node sends the array directly to all other processes:

```c
for (int i = 0; i < size; i++)
    if (i != root)
        MPI_Send(buffer, count, datatype, i, 0, comm);
```

and each of the other processes simply receive the array from the root.

```c
MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
```

#### Tree

In the tree approach, the root process sends the array to *p/2*, then recursively
they all continue sending until all processes have the data.

First, each process finds its place in the tree and determines its parent:

```c
int left = 0, right = size, parent;
while (left != rank) {
    int mid = (right + left) / 2;
    if (rank < mid) {
        right = mid;
    } else {
        parent = left;
        left = mid;
    }
}
```

Then, the processes except the root, receive the array from the determined parent:

```c
MPI_Recv(buffer, count, datatype, parent, 0, comm, MPI_STATUS_IGNORE);
```

Finally, all nodes forward the array to *p/2*, *p/4*, ...

```c
while (1) {
    int dest = (right + left) / 2;
    if (dest == rank)
        break;

    MPI_Send(buffer, count, datatype, dest, 0, comm);
    right = dest;
}
```

#### Bonus

The tree approach, as one process is closer to the root, its transfer responsability
increases logarithmically.

By representing the tree as a binary one, where each node has two children nodes,
we can guarantee that they are responsible to send arrays of at most `2*n` in length (O(n) complexity).
We can represent the binary tree as an array, where the node at index `i` has its children
at indices `2*i + 1` and `2*i + 2`. Each node `i` has its parent at index `(i-1) / 2`.

Therefore, each process receives the array from its parent:

```c
MPI_Recv(buffer, count, datatype, (rank-1)/2, 0, comm, MPI_STATUS_IGNORE);
```

and it sends it to its children:

```c
for (int i = 1; i <= 2; i++)
    if (rank*2+i < size)
        MPI_Send(buffer, count, datatype, rank*2+i, 0, comm);
```

#### MPI_Bcast

This is MPI's implementation of broadcast. We are simply calling it to
compare our implementation's performance to it.

### Performance analysis

The root process records the time necessary for the whole broadcast. We first synchronize
all the processes using `MPI_Barrier`, we start counting time and at the end of the
broadcast method, we call the barrier again so that we measure the time after all
processes have finished the procedure. This is repeated for all broadcast approaches.

```c
for (int i = 0; i < cases; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bcasts[i].func(v, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double stop = MPI_Wtime();

    bcasts[i].duration = stop - start;
    if (!check_array(v, n))
        printf("For rank=%d; %s failed", rank, bcasts[i].name);
}
```

Finally, the root process reports the recorded times, array length and bandwidths.

```c
for (int i = 0; i < cases; i++)
    printf("Longest Time for %s was %lf seconds; array=%u; bwidth=%lf B/s\n",
        bcasts[i].name, bcasts[i].duration, n, (double)n * sizeof(*v) / bcasts[i].duration);
```

The analysis reports the following:

64 processes, array size: 100000000 doubles:

Approach | Time (s) | Bandwith (B/s)
--- | --- | ---
Naive | 19.347412 | 2646348736
Tree | 1.186976 | 43134824832
Bonus | 1.894390 | 27027168128
MPI_Bcast | 0.639517 | 80060411968

128 processes, array size: 100000000 doubles:

Approach | Time (s) | Bandwith (B/s)
--- | --- | ---
Naive | 39.220236 | 2610897024
Tree | 1.441020 | 71060775680
Bonus | 2.112662 | 48469659520
MPI_Bcast | 0.723280 | 141577268352`

The results can also be found in the `run64.txt` and `run128.txt` output files.

## Part 4: Conjugate Gradient

In this problem  we solve the Laplacian equation using the method of conjugate gradients. The laplacian equation is given as ![Laplace Equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/b51f1414b4dfef0f14977223726da95123d80f39)

### Algorithm
The method of conjugate gradients is an iterative method to solve a set of linear equations of the for `Ax = b`. Here is a symmetric (`A=A'`) and positive definite (`xAx > 0` for any non zero vector `x`) matrix. For the Laplace equation the vector b is a zero vector. The implementation for the algorithm requires three kinds of matrix operations.
1. 5-point finite differences stencil
2. Dot product
3. Scaling and Scaling and addition.

As the size of grid A can get particularly large we wish to parallelize using a message passing interface to allow scaling for very large problem sizes. To parallelize we divide the grid A into smaller grid sizes and apply the above listed operations on each of the smaller grid. Scaling and addition can be done directly on any matrix as this does not need any neighbouring data. For the dot product we do a dot product for each sub matrix and then accumulate it using a call to MPI_Reduce_All. For the 5 point finite difference stencil we need the data from the neighbours. For this we divide the processes into a grid topology exchange columns and rows between neighbouring processes using the cartesian coordinates.

### How to run
To the run the program clone the repo and set the env variable CODE_HOME to the home of the repo. For example:

```
export CODE_HOME=~/code/HPC_Lab
```

Next using the submit_job script to compile and submit the job via sbatch. The script requires the following arguments:

```
submit_job <PROCESS_GRID_SIZE> <MPI_TASKS> <INV_GRID_SIZE_1D> <MAX_ITR> <ERROR_LIMIT> <LOG_FILE_NAME>
```

Example:

```
# To run a job with 512x512 grid for each process and 
# a total of 256 processes for a grid size of 8192x8192
./submit_job 512 256 0.0001220703125 10000 0.001 8192_512
```

The logs are dumped into the `assig3/conjugate_gradient/logs` folder.

### Performance report
The raw results for the performance tests can be found under `assig3/conjugate_gradient/results.data`. We performed two types of tests. In the first case we fixed the memory footprint of a single process by fixing the grid size per process (128x128) and varied the problem size from 128x128 to 16384x16384. This allows us to evaluate if the algorithm scales for huge problem sizes (given adequate resources). The results can be seen summarized in the graph below.

![Experiment 1](conjugate_gradient/images/exp1.png)

We can see a jump in time taken when the number of machines being used increases. This is understandable due to network costs.

In the second experiment we kept the problem size fixed to 8192x8192 while changing the per process size of the grid from 64x64 (16384 processes) to 8192x8192 (1 process). This allows us to evaluate the performance with changing parallelism for the same problem size.

![Experiment 2](conjugate_gradient/images/exp2.png)

Performance gain is almost linear at bigger process grid sizes. As the process grid size becomes smaller the gain becomes lesser due to the overhead in communication ie. efficiency drops down. Computing the efficieny between linear (process grid size 8192x8192) and process grid size 512x512 is:

```
(time_in_parallel/time_in_serial)/256 = 23%
```

Comparing this with the speedup we get for linear vs process grid size of 64

```
(time_in_parallel/time_in_serial)/16384 = 7%
```

So speedup decreases as the number of processes become very large (in the above scenario 128 nodes were running 16384 tasks).
