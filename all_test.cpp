#include "tensor_testing_suit.hpp"
#include "benchmark.hpp"

int main()
{
	tensor_testing_suit::RUN_ALL_TESTS();
	benchmark::RUN_ALL();
}