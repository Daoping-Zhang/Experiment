/*
 * hello_mpi.c — Demo 1: one executable, many processes.
 *
 * Launch the SAME binary with
 *
 *     mpiexec -n P ./bin/hello_mpi
 *
 * and the MPI runtime starts P independent copies of this one process.
 * Every copy runs the same code; each one finds out "who am I" (rank) and
 * "how many of us are there" (size) through MPI_COMM_WORLD.
 *
 * Run:  mpiexec -n 1 ./bin/hello_mpi
 *       mpiexec -n 4 ./bin/hello_mpi
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);          /* every MPI program starts here */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* which process am I? */
    MPI_Comm_size(MPI_COMM_WORLD, &size);   /* how many of us are there? */

    printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();                  /* every MPI program ends here */
    return 0;
}
