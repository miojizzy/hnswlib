#include <benchmark/benchmark.h>

#include "benchmarks.h"

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    
    RegisterHnswBenchmarks();
    
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    
    return 0;
}