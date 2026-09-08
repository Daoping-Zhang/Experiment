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

    MPI_Scatter(sendbuf, 1, MPI_INT, &mine, 1, MPI_INT, 0,
                MPI_COMM_WORLD);

    printf("Rank %d received %d\n", rank, mine);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
