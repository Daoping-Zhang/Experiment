/* allreduce_benchmark.c — measure REAL MPI_Allreduce.
 *
 *   mpirun -n 4 ./allreduce_benchmark <int32 elements>
 *
 * Per measured repetition each rank times its own MPI_Allreduce with
 * MPI_Wtime; the SLOWEST rank decides the collective (MPI_Reduce MPI_MAX to
 * root). Warm-up repetitions are never included. Rank 0 reports the median
 * of the measured (per-repetition, max-over-ranks) times.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define WARMUP 3
#define RUNS   10

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_d);
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = argc > 1 ? atoi(argv[1]) : 1024;
    if (n < 1) n = 1;

    int *send = malloc(sizeof(int) * n);
    int *recv = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) send[i] = rank + 1;

    for (int w = 0; w < WARMUP; w++)                 /* warm-up, untimed */
        MPI_Allreduce(send, recv, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    double times[RUNS];
    for (int r = 0; r < RUNS; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Allreduce(send, recv, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        double elapsed = t1 - t0;                    /* this rank's time */
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);                  /* slowest rank wins */
        if (rank == 0) times[r] = max_elapsed;
    }

    if (rank == 0) {
        double ms = median(times, RUNS) * 1e3;
        printf("Message Size: %lld B\n", (long long)sizeof(int) * n);
        printf("Ranks: %d\n", size);
        printf("MPI_Allreduce Time (median of %d, max-rank): %.3f ms\n",
               RUNS, ms);
    }

    free(send);
    free(recv);
    MPI_Finalize();
    return 0;
}
