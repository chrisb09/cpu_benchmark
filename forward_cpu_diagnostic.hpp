#pragma once

#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sched.h>
#include <unistd.h>
#include <mpi.h>

namespace forward_cpu_diagnostic {
inline double clock_seconds(clockid_t id) {
    timespec t{};
    if (clock_gettime(id, &t) != 0) std::abort();
    return t.tv_sec + t.tv_nsec * 1e-9;
}
struct Stamp {
    double wall, thread, process;
    static Stamp now() {
        return {clock_seconds(CLOCK_MONOTONIC), clock_seconds(CLOCK_THREAD_CPUTIME_ID),
                clock_seconds(CLOCK_PROCESS_CPUTIME_ID)};
    }
};
struct Record { std::string kind; double wall, thread, process; };
struct Buffer {
    std::vector<Record> records;
    std::string metadata;
    Buffer() {
        records.reserve(128);
        if (!std::getenv("FORWARD_DIAG_MPI_PROVENANCE")) return;
        int provided = 0, count = 0;
        std::ostringstream s;
        s << "MPI_ENV mpi_yield_when_idle="
          << (std::getenv("OMPI_MCA_mpi_yield_when_idle") ? std::getenv("OMPI_MCA_mpi_yield_when_idle") : "unset") << '\n';
        if (MPI_T_init_thread(MPI_THREAD_SINGLE, &provided) == MPI_SUCCESS) {
            MPI_T_cvar_get_num(&count);
            for (int i = 0; i < count; ++i) {
                char name[256]; int length = sizeof(name), verbosity, binding, scope;
                MPI_Datatype datatype; MPI_T_enum enumeration;
                if (MPI_T_cvar_get_info(i, name, &length, &verbosity, &datatype, &enumeration,
                                        nullptr, nullptr, &binding, &scope) != MPI_SUCCESS ||
                    std::string(name) != "mpi_yield_when_idle") continue;
                MPI_T_cvar_handle handle; int elements = 0;
                if (MPI_T_cvar_handle_alloc(i, nullptr, &handle, &elements) == MPI_SUCCESS) {
                    int value = 0;
                    if (elements == 1 && (datatype == MPI_INT || datatype == MPI_C_BOOL) &&
                        MPI_T_cvar_read(handle, &value) == MPI_SUCCESS)
                        s << "MPI_EFFECTIVE mpi_yield_when_idle=" << value << " source=MPI_T\n";
                    MPI_T_cvar_handle_free(&handle);
                }
            }
            MPI_T_finalize();
        }
        metadata = s.str();
    }
    ~Buffer() {
        const char* dir = std::getenv("FORWARD_DIAG_DIR");
        if (!dir || records.empty()) return;
        const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
        std::ofstream out(std::string(dir) + "/rank_" + (rank ? rank : "unknown") +
                          "_pid_" + std::to_string(getpid()) + ".txt", std::ios::app);
        out.precision(12);
        out << metadata;
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            out << "TIMING kind=" << r.kind << " call=" << i
                << " wall_s=" << r.wall << " thread_s=" << r.thread
                << " process_s=" << r.process << '\n';
        }
    }
};
inline Buffer& buffer() { static Buffer b; return b; }
inline void record(const char* kind, Stamp start, Stamp end) {
    buffer().records.push_back({kind, end.wall-start.wall, end.thread-start.thread,
                                end.process-start.process});
}
template<class Tensor, class Model>
void metadata(const char* kind, const Tensor& input, const Model& model, int intra, int inter) {
    auto& b = buffer();
    if (b.metadata.find("META kind=") != std::string::npos) return;
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (sched_getaffinity(0, sizeof(cpus), &cpus) != 0) std::abort();
    std::ostringstream s;
    s << "META kind=" << kind << " intra=" << intra << " inter=" << inter
      << " shape=" << input.sizes() << " strides=" << input.strides()
      << " dtype=" << input.scalar_type() << " device=" << input.device()
      << " contiguous=" << input.is_contiguous() << " affinity=";
    for (int i=0; i<CPU_SETSIZE; ++i) if (CPU_ISSET(i, &cpus)) s << i << ',';
    for (const auto& p : model.parameters()) {
        s << " model_dtype=" << p.scalar_type() << " model_device=" << p.device();
        break;
    }
    s << '\n';
    b.metadata += s.str();
}
template<class Function>
auto forward(const char* kind, Function&& function) {
    buffer();
    const auto start = Stamp::now();
    auto result = function();
    const auto end = Stamp::now();
    record(kind, start, end);
    return result;
}
}
