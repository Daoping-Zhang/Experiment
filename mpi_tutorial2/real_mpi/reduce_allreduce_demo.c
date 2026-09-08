/* reduce_allreduce_demo.c — "what real MPI code looks like".
 *
 * Each rank owns a small local value; we call MPI_Reduce (SUM -> root 0)
 * and MPI_Allreduce (SUM -> everyone), exactly like the MiniMPI teaching
 * collectives — except MPI itself implements them.
 *
 * Build:  make
 * Run:    mpirun -n 4 ./reduce_allreduce_demo
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local = rank + 1;              /* rank 0..3 -> 1,2,3,4 */

    int reduced = 0;
    MPI_Reduce(&local, &reduced, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("REDUCE root result = %d\n", reduced);

    int all_result = 0;
    MPI_Allreduce(&local, &all_result, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    printf("Rank %d AllReduce result = %d\n", rank, all_result);

    fflush(stdout);
    MPI_Finalize();
    return 0;
}
