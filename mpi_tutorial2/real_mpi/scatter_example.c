/* scatter_example.c — one collective, one file: MPI_Scatter.
 *
 *   mpirun -n 4 ./scatter_example
 *
 * Rank 0 owns the array [10,20,30,40]; MPI_Scatter sends one element of it
 * to each rank, in rank order (root keeps its own part).
 * This maps to GEMM: scatter rows of matrix A to ranks.
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[4] = {10, 20, 30, 40};   /* only rank 0 uses this */
    int mine = -1;

    /* MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount,
     *              recvtype, root, comm)
     *   sendbuf   : root's array to split [10,20,30,40]
     *   sendcount : 1 — elements each rank receives
     *   sendtype  : MPI_INT
     *   recvbuf   : &mine — this rank's one piece
     *   recvcount : 1
     *   recvtype  : MPI_INT
     *   root      : 0
     *   comm      : MPI_COMM_WORLD
     */
    MPI_Scatter(
        sendbuf,          /* sendbuf  (root only) */
        1,                /* sendcount: one element per rank */
        MPI_INT,          /* sendtype */
        &mine,            /* recvbuf: this rank's piece */
        1,                /* recvcount */
        MPI_INT,          /* recvtype */
        0,                /* root */
        MPI_COMM_WORLD    /* communicator */
    );

    printf("Rank %d received %d\n", rank, mine);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
