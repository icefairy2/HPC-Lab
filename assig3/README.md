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

### Performance analysis

Each process reports the time it spent on receiving and forwarding the array for each
of the four approach. Then, the root process gathers the information and holds the
longest time for each approach. It then reports the time, array length and bandwidth
(array size / time) for each of the approaches.

TBA
