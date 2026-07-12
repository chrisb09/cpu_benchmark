#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {
void log_line(const char *message, int rank = -1, int size = -1, const char *host = nullptr)
{
    if (rank >= 0) {
        std::fprintf(stdout, "[probe] %s rank=%d size=%d host=%s\n", message, rank, size, host ? host : "?");
    } else {
        std::fprintf(stdout, "[probe] %s\n", message);
    }
    std::fflush(stdout);
}
}

int main(int argc, char **argv)
{
    const char *role = argc > 1 ? argv[1] : "unknown";
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    std::fprintf(stdout, "[probe] before MPI_Init role=%s host=%s slurm_procid=%s\n",
                 role, host, std::getenv("SLURM_PROCID") ? std::getenv("SLURM_PROCID") : "?");
    std::fflush(stdout);

    const int init_rc = MPI_Init(&argc, &argv);
    if (init_rc != MPI_SUCCESS) {
        std::fprintf(stderr, "[probe] MPI_Init failed role=%s rc=%d\n", role, init_rc);
        std::fflush(stderr);
        return init_rc;
    }

    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    log_line("after MPI_Init", rank, size, host);

    const int color = std::strcmp(role, "solver") == 0 ? 0 : MPI_UNDEFINED;
    log_line("before MPI_Comm_split", rank, size, host);
    MPI_Comm local_comm = MPI_COMM_NULL;
    const int split_rc = MPI_Comm_split(MPI_COMM_WORLD, color, rank, &local_comm);
    if (split_rc != MPI_SUCCESS) {
        std::fprintf(stderr, "[probe] MPI_Comm_split failed rank=%d rc=%d\n", rank, split_rc);
        MPI_Abort(MPI_COMM_WORLD, split_rc);
    }

    if (local_comm != MPI_COMM_NULL) {
        int local_rank = -1;
        int local_size = -1;
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_size(local_comm, &local_size);
        std::fprintf(stdout, "[probe] after MPI_Comm_split role=%s rank=%d local_rank=%d local_size=%d host=%s\n",
                     role, rank, local_rank, local_size, host);
        std::fflush(stdout);
        MPI_Comm_free(&local_comm);
    } else {
        log_line("after MPI_Comm_split", rank, size, host);
    }

    log_line("before world barrier", rank, size, host);
    MPI_Barrier(MPI_COMM_WORLD);
    log_line("after world barrier", rank, size, host);

    MPI_Finalize();
    std::fprintf(stdout, "[probe] after MPI_Finalize role=%s rank=%d host=%s\n", role, rank, host);
    std::fflush(stdout);
    return 0;
}
