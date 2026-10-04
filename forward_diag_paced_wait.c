#include <mpi.h>
#include <time.h>
#include <errno.h>

/* Diagnostic-only PMPI interposition, preloaded into solver processes only. */
int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]) {
    int complete = 0;
    while (!complete) {
        int rc = PMPI_Testall(count, requests, &complete, statuses);
        if (rc != MPI_SUCCESS || complete) return rc;
        struct timespec delay = {0, 100000};
        while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
    }
    return MPI_SUCCESS;
}
