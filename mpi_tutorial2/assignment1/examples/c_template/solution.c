/* c_template/solution.c — Assignment 1 SKELETON (fill in the TODOs only).
 *
 * Compiled by build.sh with mpicc and launched by run.sh under:
 *     mpirun -n P ./run.sh <input.txt>
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * TODO 1:
     * Read input and obtain this rank's Local Vector.
     */

    /*
     * TODO 2:
     * Perform Reduce SUM to Rank 0.
     */

    /*
     * TODO 3:
     * Perform AllReduce SUM to every Rank.
     */

    /*
     * TODO 4:
     * Print output using the required format.
     */

    MPI_Finalize();
    return 0;
}
