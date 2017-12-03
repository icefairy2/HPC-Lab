# Assignment 3

## Broadcast

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
Naive | 19.347412 | 41349199
Tree | 1.186976 | 673981638
Bonus | 1.894390 | 422299502
MPI_Bcast | 0.639517 | 1250943937

128 processes, array size: 100000000 doubles:

Approach | Time (s) | Bandwith (B/s)
--- | --- | ---
Naive | 39.220236 | 20397633
Tree | 1.441020 | 555162310
Bonus | 2.112662 | 378669215
MPI_Bcast | 0.723280 | 1106072409

The results can also be found in the `run64.txt` and `run128.txt` output files.
