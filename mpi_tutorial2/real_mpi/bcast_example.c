/* bcast_example.c — one collective, one file: MPI_Bcast.
 *
 *   mpirun -n 4 ./bcast_example
 *
 * Rank 0 initially owns value = 42; every other rank starts with 0.
 * MPI_Bcast gives all ranks the same value from rank 0.
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = (rank == 0) ? 42 : 0;

    printf("Before Bcast  Rank %d value = %d\n", rank, value);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("After Bcast   Rank %d value = %d\n", rank, value);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
