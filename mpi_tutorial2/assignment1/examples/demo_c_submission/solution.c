/* demo_c_submission — MPI Reduce + AllReduce (SUM, root = 0). */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int *read_local(const char *path, int rank, int *n_out) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int P, N, r;
    if (fscanf(fp, "%d %d", &P, &N) != 2) { fprintf(stderr, "bad header\n"); exit(1); }
    int *row = malloc(sizeof(int) * N);
    for (r = 0; r < P; r++) {
        int i;
        for (i = 0; i < N; i++) {
            int v;
            if (fscanf(fp, "%d", &v) != 1) { fprintf(stderr, "bad row\n"); exit(1); }
            if (r == rank) row[i] = v;
        }
    }
    fclose(fp);
    *n_out = N;
    return row;
}

static void print_vec(const char *prefix, int rank, const int *v, int n) {
    int i;
    printf("%s rank=%d: ", prefix, rank);
    for (i = 0; i < n; i++) printf("%s%d", i ? " " : "", v[i]);
    printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 2) { fprintf(stderr, "usage: solution <input.txt>\n"); MPI_Abort(MPI_COMM_WORLD, 2); }

    int N;
    int *local = read_local(argv[1], rank, &N);
    int *reduced = rank == 0 ? malloc(sizeof(int) * N) : NULL;
    MPI_Reduce(local, reduced, N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) print_vec("REDUCE", 0, reduced, N);

    int *allv = malloc(sizeof(int) * N);
    MPI_Allreduce(local, allv, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    print_vec("ALLREDUCE", rank, allv, N);

    free(local); free(allv); if (reduced) free(reduced);
    MPI_Finalize();
    return 0;
}
