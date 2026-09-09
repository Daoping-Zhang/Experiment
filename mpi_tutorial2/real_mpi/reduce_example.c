/* reduce_example.c — one collective, one file: MPI_Reduce.
 *
 *   mpirun -n 4 ./reduce_example
 *
 * local = rank + 1  (1,2,3,4); MPI_Reduce(SUM) puts 10 only on rank 0.
 * Reduce result exists only at the root.
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

    /* MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm)
     *   sendbuf : &local  — this rank's value (all ranks send)
     *   recvbuf : &result — only meaningful on root
     *   count   : 1
     *   datatype: MPI_INT
     *   op      : MPI_SUM
     *   root    : 0       — only this rank receives the result
     *   comm    : MPI_COMM_WORLD
     */
    MPI_Reduce(
        &local,           /* sendbuf */
        &result,          /* recvbuf (root only) */
        1,                /* count */
        MPI_INT,          /* datatype */
        MPI_SUM,          /* op */
        0,                /* root */
        MPI_COMM_WORLD    /* communicator */
    );

    if (rank == 0)
        printf("Rank 0 Reduce result = %d\n", result);

    fflush(stdout);
    MPI_Finalize();
    return 0;
}
