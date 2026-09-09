/* gather_example.c — one collective, one file: MPI_Gather.
 *
 *   mpirun -n 4 ./gather_example
 *
 * Every rank owns a local value; MPI_Gather collects them on rank 0 in
 * rank order: 10 20 30 40.
 * This maps to GEMM: gather the local C pieces into the final C on rank 0.
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local = (rank + 1) * 10;         /* 10, 20, 30, 40 */
    int recvbuf[4] = {0, 0, 0, 0};       /* only rank 0 uses this */

    /* MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
     *             recvtype, root, comm)
     *   sendbuf   : &local — this rank's one value
     *   sendcount : 1
     *   sendtype  : MPI_INT
     *   recvbuf   : recvbuf — root collects [10,20,30,40] here
     *   recvcount : 1 — elements per rank (not total)
     *   recvtype  : MPI_INT
     *   root      : 0
     *   comm      : MPI_COMM_WORLD
     */
    MPI_Gather(
        &local,           /* sendbuf: this rank's value */
        1,                /* sendcount */
        MPI_INT,          /* sendtype */
        recvbuf,          /* recvbuf (root only) */
        1,                /* recvcount: one element per rank */
        MPI_INT,          /* recvtype */
        0,                /* root */
        MPI_COMM_WORLD    /* communicator */
    );

    if (rank == 0) {
        printf("Gathered result:");
        for (int i = 0; i < size; i++) printf(" %d", recvbuf[i]);
        printf("\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
