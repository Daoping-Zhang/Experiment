/*
 * ping_pong.c — Demo 3 / Exercise 2: a message goes back and forth.
 *
 * Rank 0 sends 42 to rank 1, rank 1 adds 1 to its OWN copy and sends it
 * back, rank 0 receives 43 and prints it.
 *
 * Two things this shows:
 *   1. communication is two-way (send -> recv -> send -> recv);
 *   2. every process owns its own local variable — there is no shared
 *      memory, so "value + 1" on rank 1 does not touch rank 0's value.
 *
 * Run:  mpiexec -n 2 ./bin/ping_pong
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int value = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("This example requires at least 2 MPI processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        value = 42;
        printf("Rank 0 sent %d to rank 1\n", value);
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);   /* 0 -> 1 */

        MPI_Recv(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);                          /* 1 -> 0 */
        printf("Rank 0 received %d\n", value);
    } else if (rank == 1) {
        MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("Rank 1 received %d\n", value);

        value = value + 1;   /* rank 1 changes its own local copy */
        printf("Rank 1 sent back %d\n", value);
        MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
