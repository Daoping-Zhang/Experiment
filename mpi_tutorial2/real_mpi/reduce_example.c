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

    MPI_Reduce(&local, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Rank 0 Reduce result = %d\n", result);

    fflush(stdout);
    MPI_Finalize();
    return 0;
}
