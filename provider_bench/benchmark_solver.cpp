#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mpi.h>
#include <unistd.h>
#include <sys/resource.h>

#include "ml_coupling.hpp"
#ifdef WITH_AIX
#include "provider/ml_coupling_provider_aixelerator.hpp"
#endif
#include "provider/ml_coupling_provider_smartsim.hpp"
#include "provider/ml_coupling_provider_phydll.hpp"
#include "application/ml_coupling_application.hpp"
#include "behavior/ml_coupling_behavior_default.hpp"
#include "normalization/ml_coupling_minmax_normalization.hpp"

// Dummy Application
template <typename In, typename Out>
class BenchmarkApplication : public MLCouplingApplication<In, Out> {
public:
    BenchmarkApplication(MLCouplingData<In> input_data, MLCouplingData<Out> output_data)
        : MLCouplingApplication<In, Out>(std::move(input_data), std::move(output_data), new MLCouplingMinMaxNormalization<In, Out>(0.0f, 1.0f, 0.0f, 1.0f)) {}
protected:
    MLCouplingData<In> preprocess(MLCouplingData<In> input_data) override { return input_data; }
    void coupling_step(MLCouplingData<In>) override {}
    MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data) override { return output_data; }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Create solver-only communicator (DL client ranks use different color)
    MPI_Comm solver_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &solver_comm);

    if (solver_comm != MPI_COMM_NULL) {
        MPI_Comm_size(solver_comm, &world_size);
    }

    std::string provider = "AIX";
    std::string model_path = "";
    std::string schema = "mini_app";
    int total_inputs = 100000;
    int batch_size = 0;
    int min_batch_size = 0;
    int min_batch_timeout = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--provider" && i + 1 < argc) provider = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--schema" && i + 1 < argc) schema = argv[++i];
        else if (arg == "--inputs" && i + 1 < argc) total_inputs = std::stoi(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--min-batch-size" && i + 1 < argc) min_batch_size = std::stoi(argv[++i]);
        else if (arg == "--min-batch-timeout" && i + 1 < argc) min_batch_timeout = std::stoi(argv[++i]);
    }

    int inputs_per_rank = total_inputs / world_size;
    if (world_rank == world_size - 1) {
        inputs_per_rank += total_inputs % world_size;
    }

    int current_bs = inputs_per_rank;
    int num_batches = 1;

    // We can also simulate batches if needed, but for now we just process everything in 1 batch.
    // If the benchmark is for smartsim server-side batching, we just send all our data at once.

    std::vector<int> in_shape, out_shape;
    size_t in_size = 0, out_size = 0;
    if (schema == "mmcp") {
        in_shape = {current_bs * 3, 10, 512};
        in_size = current_bs * 3 * 10 * 512;
        out_shape = {current_bs * 3, 2, 512};
        out_size = current_bs * 3 * 2 * 512;
    } else {
        in_shape = {current_bs, 18};
        in_size = current_bs * 18;
        out_shape = {current_bs}; // Standard model output is 1D tensor [bs]
        out_size = current_bs;
    }

    std::vector<float> in_buffer(in_size, 1.0f);
    std::vector<float> out_buffer(out_size, 0.0f);

    MLCouplingData<float> input_data{std::vector<MLCouplingTensor<float>>{
        MLCouplingTensor<float>::wrap_flat(in_buffer.data(), in_shape, MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal)
    }};
    MLCouplingData<float> output_data{std::vector<MLCouplingTensor<float>>{
        MLCouplingTensor<float>::wrap_flat(out_buffer.data(), out_shape, MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal)
    }};

    MLCouplingProvider<float, float>* prov = nullptr;
    if (provider == "AIX") {
#ifdef WITH_AIX
        prov = new MLCouplingProviderAixelerator<float, float>(model_path, current_bs, solver_comm, false);
#else
        if (world_rank == 0) std::cerr << "AIX provider not compiled into this benchmark_solver build." << std::endl;
        MPI_Finalize();
        return 1;
#endif
    } else if (provider == "PHYDLL") {
        prov = new MLCouplingProviderPhydll<float, float>(model_path, "TORCH", "CPU");
    } else if (provider == "SMARTSIM") {
        std::string m_name = "benchmark_model";
        if (std::getenv("MLCOUPLING_MULTI_MODEL") != nullptr) {
            m_name += "_" + std::to_string(world_rank);
        }
        prov = new MLCouplingProviderSmartsim<float, float>("CPU", "TORCH", model_path, m_name, "", -1, 1, 0, 0, batch_size, min_batch_size, min_batch_timeout, 2000, 2000, 2000000);
    } else {
        if (world_rank == 0) std::cerr << "Unknown provider: " << provider << std::endl;
        MPI_Finalize();
        return 1;
    }

    // NEW: Print how the ranks split the data
    if (world_rank == 0 || world_rank == world_size - 1 || world_rank == 33) {
        std::cout << "Rank " << world_rank << " split -> inputs_per_rank=" << inputs_per_rank << " | current_bs=" << current_bs << std::endl;
    }

    if (world_rank == 0) {
        std::cout << "Provider: " << provider << " | Inputs/rank: " << inputs_per_rank 
                  << " | Batches: " << num_batches << " | BS: " << current_bs;
        if (provider == "SMARTSIM") {
            std::cout << " | RedisAI batch_size=" << batch_size 
                      << " min_batch_size=" << min_batch_size 
                      << " min_batch_timeout=" << min_batch_timeout;
        }
        std::cout << std::endl;
    }

    double local_cold_elapsed = 0.0;
    double local_warm_elapsed = 0.0;
    double max_cold_elapsed = 0.0;
    double max_warm_elapsed = 0.0;
    double mem_mb = 0.0;
    double sum_mem = 0.0;

    // Destroy the coupling before MPI_Finalize so the PhyDLL provider can
    // signal and finalize the DL ranks while MPI is still available.
    {
        auto* app = new BenchmarkApplication<float, float>(input_data, output_data);
        auto* beh = new MLCouplingBehaviorDefault();
        MLCoupling<float, float> coupling(prov, app, beh, CouplingType::STATIC, &(app->input_data_after_preprocessing), &(app->output_data_before_postprocessing));

        MPI_Barrier(solver_comm);

        try {
            if (world_rank == 0) std::cout << "Starting cold inference (iteration 0)..." << std::endl;
            
            auto start_cold = std::chrono::high_resolution_clock::now();
            coupling.ordered()
                .set(input_data)
                .inference()
                .get(output_data);
            auto end_cold = std::chrono::high_resolution_clock::now();
            local_cold_elapsed = std::chrono::duration<double>(end_cold - start_cold).count();
            
            if (world_rank == 0) std::cout << "Starting warm inference (iteration 1)..." << std::endl;
            
            auto start_warm = std::chrono::high_resolution_clock::now();
            coupling.ordered()
                .set(input_data)
                .inference()
                .get(output_data);
            auto end_warm = std::chrono::high_resolution_clock::now();
            local_warm_elapsed = std::chrono::duration<double>(end_warm - start_warm).count();
            
            if (world_rank == 0) std::cout << "Inference iterations complete!" << std::endl;
        } catch(const std::exception& e) {
            std::cerr << "Rank " << world_rank << " Exception: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            MPI_Abort(solver_comm, 1);
        }

        MPI_Barrier(solver_comm);
        struct rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
        mem_mb = r_usage.ru_maxrss / 1024.0;
        
        MPI_Reduce(&local_cold_elapsed, &max_cold_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, solver_comm);
        MPI_Reduce(&local_warm_elapsed, &max_warm_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, solver_comm);
    }
    
    MPI_Reduce(&mem_mb, &sum_mem, 1, MPI_DOUBLE, MPI_SUM, 0, solver_comm);

    if (world_rank == 0) {
        std::cout << "RESULT:" << max_cold_elapsed << "," << max_warm_elapsed << "," << sum_mem << std::endl;
    }

    MPI_Finalize();
    return 0;
}
