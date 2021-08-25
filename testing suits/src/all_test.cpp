#include "../inc/tests/tensor_testing_suit.hpp"
#include "../inc/benchmarks/benchmark.hpp"

int main()
{
	tensor_testing_suit::RUN_ALL_TESTS();
	benchmark::RUN_ALL();
}