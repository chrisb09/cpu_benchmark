#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <unistd.h>
#include <sys/resource.h>

#include "ml_coupling.hpp"
#include "provider/ml_coupling_provider_aixelerator.hpp"
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
    MLCouplingData<Out> ml_step(MLCouplingData<In>) override { return this->output_data_before_postprocessing; }
    MLCouplingData<Out> postprocess(MLCouplingData<Out> output_data) override { return output_data; }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::string provider = "AIX";
    std::string model_path = "";
    std::string schema = "mini_app";
    int total_inputs = 100000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--provider" && i + 1 < argc) provider = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--schema" && i + 1 < argc) schema = argv[++i];
        else if (arg == "--inputs" && i + 1 < argc) total_inputs = std::stoi(argv[++i]);
    }

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int inputs_per_rank = total_inputs / world_size;
    if (world_rank == world_size - 1) {
        inputs_per_rank += total_inputs % world_size;
    }

    int max_bs = (schema == "mmcp") ? 5000 : 100000;
    int current_bs = std::min(inputs_per_rank, max_bs);
    if (current_bs == 0) current_bs = 1;

    int num_batches = (inputs_per_rank + current_bs - 1) / current_bs;

    std::vector<int> in_shape, out_shape;
    size_t in_size = 0, out_size = 0;
    if (schema == "mmcp") {
        in_shape = {5, current_bs, 512};
        in_size = 5 * current_bs * 512;
        out_shape = {5, 2, 512}; // Model outputs [5, 2, 512] regardless of batch size
        out_size = 5 * 2 * 512;
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
        prov = new MLCouplingProviderAixelerator<float, float>(model_path, 1, MPI_COMM_WORLD, false);
    } else if (provider == "PHYDLL") {
        prov = new MLCouplingProviderPhydll<float, float>(model_path, "TORCH", "CPU");
    } else if (provider == "SMARTSIM") {
        prov = new MLCouplingProviderSmartsim<float, float>("CPU", "TORCH", model_path, "benchmark_model", "", -1, 1, 0, 0, 1, 0, 0, 900, 900, 900000);
    } else {
        if (world_rank == 0) std::cerr << "Unknown provider: " << provider << std::endl;
        MPI_Finalize();
        return 1;
    }

    auto* app = new BenchmarkApplication<float, float>(input_data, output_data);
    auto* beh = new MLCouplingBehaviorDefault();

    MLCoupling<float, float> coupling(prov, app, beh, CouplingType::STATIC, &(app->input_data_after_preprocessing), &(app->output_data_before_postprocessing));


    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    try {
        if (world_rank == 0) std::cout << "Starting inference loop for " << num_batches << " batches..." << std::endl;
        for (int b = 0; b < num_batches; ++b) {
            if (world_rank == 0 && b == 0) std::cout << "  -> Batch 0 starting..." << std::endl;
            coupling.ordered()
                .set(input_data)
                .inference()
                .get(output_data);
            if (world_rank == 0 && b == 0) std::cout << "  -> Batch 0 complete!" << std::endl;
        }
    } catch(const std::exception& e) {
        std::cerr << "Rank " << world_rank << " Exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    double mem_mb = r_usage.ru_maxrss / 1024.0;

    double local_elapsed = elapsed.count();
    double max_elapsed = 0.0;
    double sum_mem = 0.0;
    
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mem_mb, &sum_mem, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        std::cout << "RESULT:" << max_elapsed << "," << sum_mem << std::endl;
    }

    MPI_Finalize();
    return 0;
}
