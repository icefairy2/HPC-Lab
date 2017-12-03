#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int naive_bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        for (int i = 0; i < size; i++)
            if (i != root)
                MPI_Send(buffer, count, datatype, i, 0, comm);
    } else {
        MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int tree_bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

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

    if (rank != root)
        MPI_Recv(buffer, count, datatype, parent, 0, comm, MPI_STATUS_IGNORE);

    while (1) {
        int dest = (right + left) / 2;
        if (dest == rank)
            break;

        MPI_Send(buffer, count, datatype, dest, 0, comm);
        right = dest;
    }

    return MPI_SUCCESS;
}

int bonus_bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank != root)
        MPI_Recv(buffer, count, datatype, (rank-1)/2, 0, comm, MPI_STATUS_IGNORE);

    for (int i = 1; i <= 2; i++)
        if (rank*2+i < size)
            MPI_Send(buffer, count, datatype, rank*2+i, 0, comm);

    return MPI_SUCCESS;
}

struct bcast {
    int (*func)(void*, int, MPI_Datatype, int, MPI_Comm);
    char* name;
    double duration;
};

double* init_array(const int n)
{
    double* v = malloc(n * sizeof(*v));
    for (int i = 0; i < n; i++)
        v[i] = i;
    return v;
}

int check_array(const double* v, const int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] != i)
            return 0;
    return 1;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    const int n = 100000000;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double* v = rank == 0 ? init_array(n) : calloc(sizeof(*v), n);

    struct bcast bcasts[] = {
        { naive_bcast, "Naive" },
        { tree_bcast, "Tree" },
        { bonus_bcast, "Bonus" },
        { MPI_Bcast, "MPI_Bcast" }
    };
    const int cases = sizeof(bcasts) / sizeof(struct bcast);

    if (rank == 0)
        printf("%d cases; %d processes\n", cases, size);

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

    if (rank == 0)
        for (int i = 0; i < cases; i++)
            printf("Longest Time for %s was %lf seconds; array=%u; bwidth=%lf B/s\n",
                bcasts[i].name, bcasts[i].duration, n, (double)n * sizeof(*v) / bcasts[i].duration);

    free(v);
    MPI_Finalize();
    return 0;
}
