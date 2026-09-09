/* barrier_example.c — one collective, one file: MPI_Barrier.
 *
 *   mpirun -n 4 ./barrier_example
 *
 * MPI_Barrier is synchronization, not data reduction: every rank waits
 * until all ranks reached the barrier, then all continue. We never claim
 * anything about how MPI implements the barrier internally.
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Rank %d before barrier\n", rank);
    fflush(stdout);

    /* MPI_Barrier(comm)
     *   comm : MPI_COMM_WORLD
     * Blocking synchronization: no rank proceeds past this call until
     * every rank in the communicator has reached it. No data is moved —
     * this is NOT a reduction.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Rank %d after barrier\n", rank);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
