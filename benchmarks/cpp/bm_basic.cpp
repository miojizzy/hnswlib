#include <benchmark/benchmark.h>
#include <vector>
#include <random>

#include "hnswlib/hnswalg.h"

// hnsw build benchmark
static void BM_HnswBuildDataSizeBench(benchmark::State& state) { 
    const size_t dim = 128;
    size_t x_data_size = state.range(0);
    
    std::mt19937 rng(42); // 固定种子以确保可重复性
    std::vector< std::array<float, dim> > embeddings(x_data_size);
    for(auto& emb: embeddings) {
        std::generate(emb.begin(), emb.end(), rng);
    }

    for (auto _: state) {
        auto space = std::make_shared<hnswlib::InnerProductSpace>(dim);
        auto index = std::make_shared<hnswlib::HierarchicalNSW<float>>(
            space.get(), x_data_size*2);
        for (size_t i = 0; i < x_data_size; i++) {
            auto& emb = embeddings[i];
            index->addPoint(emb.data(), i);
        }
        benchmark::DoNotOptimize(index);
    }

    state.SetComplexityN(state.range(0));
}

void RegisterHnswBenchmarks() {
    BENCHMARK(BM_HnswBuildDataSizeBench)
        ->Range( 1 << 9 /* 512 */, 1 << 18 /* 262144 */);

}