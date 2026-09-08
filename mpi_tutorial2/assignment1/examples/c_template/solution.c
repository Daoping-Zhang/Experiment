/* c_template/solution.c — Assignment 1 SKELETON (fill in the TODOs).
 *
 * Run by the grader: mpirun -n P ./run.sh <input.txt>  (after build.sh).
 * Allowed: MPI_Reduce / MPI_Allreduce, or your own Send/Recv design.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        fprintf(stderr, "usage: solution <input.txt>\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* TODO 1 — read THIS rank's local vector from argv[1]:
     *   first line "P N", then P lines (line r = vector of rank r).
     * int N; int *local = ...; */
    int N = 0;
    int *local = NULL;

    /* TODO 2 — Reduce (SUM -> root 0). Only rank 0 prints:
     *   REDUCE rank=0: v0 v1 ... vN-1                          */
    int *reduced = NULL;
    /* MPI_Reduce(...); */

    if (rank == 0) {
        int i;
        printf("REDUCE rank=0: ");
        for (i = 0; i < N; i++) printf("%s%d", i ? " " : "", reduced[i]);
        printf("\n");
        fflush(stdout);
    }

    /* TODO 3 — AllReduce (SUM -> every rank). Every rank prints:
     *   ALLREDUCE rank=<rank>: v0 v1 ... vN-1                  */
    int *all_out = NULL;
    /* MPI_Allreduce(...); */

    {
        int i;
        printf("ALLREDUCE rank=%d: ", rank);
        for (i = 0; i < N; i++) printf("%s%d", i ? " " : "", all_out[i]);
        printf("\n");
        fflush(stdout);
    }

    free(local); free(reduced); free(all_out);
    MPI_Finalize();
    return 0;
}
