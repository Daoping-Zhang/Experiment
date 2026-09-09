/* allreduce_example.c — one collective, one file: MPI_Allreduce.
 *
 *   mpirun -n 4 ./allreduce_example
 *
 * local = rank + 1 (1,2,3,4); MPI_Allreduce(SUM) gives EVERY rank 10.
 * This is what MiniMPI's Recursive Doubling / Ring showed how to
 * implement — here the MPI library does it for you.
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local = rank + 1;
    int result = -1;

    /* MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm)
     *   sendbuf : &local — this rank's value (all ranks send)
     *   recvbuf : &result — EVERY rank receives the reduced result
     *   count   : 1
     *   datatype: MPI_INT
     *   op      : MPI_SUM
     *   comm    : MPI_COMM_WORLD
     * (No root argument: the result is returned to all ranks.)
     */
    MPI_Allreduce(
        &local,           /* sendbuf */
        &result,          /* recvbuf — every rank gets the result */
        1,                /* count */
        MPI_INT,          /* datatype */
        MPI_SUM,          /* op */
        MPI_COMM_WORLD    /* communicator */
    );

    printf("Rank %d AllReduce result = %d\n", rank, result);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
