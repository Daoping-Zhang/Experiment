/*
 * send_recv.c — Demo 2: two processes exchange one message explicitly.
 *
 * Rank 0 sends the value 42 to rank 1 with MPI_Send; rank 1 receives it
 * with MPI_Recv and prints it.
 *
 * Every argument of MPI_Send / MPI_Recv is written on its own line ON
 * PURPOSE: the point of this demo is to look at the raw API — buffer,
 * count, datatype, source/destination, tag, communicator — not to hide it.
 *
 * Run:  mpiexec -n 2 ./bin/send_recv
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    int value = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* This demo needs one sender and one receiver. Exit cleanly otherwise. */
    if (size < 2) {
        if (rank == 0)
            printf("This example requires at least 2 MPI processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        value = 42;   /* rank 0's own local variable */

        MPI_Send(
            &value,          /* buffer:      where the data lives      */
            1,               /* count:       how many elements         */
            MPI_INT,         /* datatype:    what each element is      */
            1,               /* destination: rank to send to           */
            0,               /* tag:         label for this message    */
            MPI_COMM_WORLD); /* communicator: which group of processes */
    } else if (rank == 1) {
        MPI_Recv(
            &value,               /* buffer:      where to put the data */
            1,                    /* count                             */
            MPI_INT,              /* datatype                          */
            0,                    /* source:      rank to receive from */
            0,                    /* tag                               */
            MPI_COMM_WORLD,       /* communicator                      */
            MPI_STATUS_IGNORE);   /* we ignore the message status      */

        printf("Rank 1 received value %d from rank 0\n", value);
    }

    MPI_Finalize();
    return 0;
}
