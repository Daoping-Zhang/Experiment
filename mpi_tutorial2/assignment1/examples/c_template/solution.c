/* c_template/solution.c — Assignment 1 (Parallel GEMM) SKELETON.
 *
 * Fill in the TODOs only — do not submit a copy of someone else's answer.
 * Build:   ./build.sh          (mpicc)
 * Run:     mpirun -n P ./run.sh <input.txt>   (outer layer starts mpirun)
 *
 * Input:
 *   M K N
 *   <M x K matrix A>
 *   <K x N matrix B>
 * Output (rank 0 only, to stdout):
 *   RESULT M N
 *   <row 0 of C>
 *   ...
 *   <row M-1 of C>
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
     * TODO 1: rank 0 reads M, K, N and matrices A (M x K) and B (K x N)
     *         from argv[1].
     *
     * TODO 2: share the problem size (and B) with every rank — Bcast is
     *         useful here.
     *
     * TODO 3: scatter rows of A to the ranks (1D row partitioning) —
     *         Scatter is useful here.
     *
     * TODO 4: each rank computes its local piece:  local_C = local_A x B.
     *
     * TODO 5: collect the local C pieces on rank 0 — Gather is useful
     *         here.
     *
     * TODO 6: rank 0 prints the final C as:
     *         RESULT M N
     *         row 0 ... row M-1
     */

    MPI_Finalize();
    return 0;
}
