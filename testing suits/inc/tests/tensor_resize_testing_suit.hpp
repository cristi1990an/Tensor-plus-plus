#pragma once

#include "../../../inc/tensor.hpp"

#include <algorithm>
#include <iostream>

namespace tensor_resize_testing_suit
{
	using namespace tensor_lib;

	void TEST_1()
	{
		tensor<int, 10> tsor;

		tsor.resize(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

		std::cout << "\tTEST 1 PASSED.\n";
	}

	void TEST_2()
	{
		tensor<int, 1> tsor;

		tsor.resize(10);

		if (tsor.order_of_current_dimension() != 10)
		{
			throw std::runtime_error("TEST_2 in 'tensor_resize_testing_suit' failed!\n");
		}
		if (tsor.size_of_current_tensor() != 10)
		{
			throw std::runtime_error("TEST_2 in 'tensor_resize_testing_suit' failed!\n");
		}

		std::cout << "\tTEST 2 PASSED.\n";
	}

	void TEST_3()
	{
		tensor<int, 5> tsor(7, 7, 7, 7, 7);

		tsor.resize(1, 2, 3, 4, 5);
		std::array<size_t, 5> expected_sizes{ 120, 120, 60, 20, 5 };
		std::array<size_t, 5> expected_ranks{ 1, 2, 3, 4, 5 };

		/*std::cout << "Ranks: ";
		for (const auto& val : tsor.get_ranks())
		{
			std::cout << val << ' ';
		}
		std::cout << std::endl;

		std::cout << "Sizes: ";
		for (const auto& val : tsor.get_sizes())
		{
			std::cout << val << ' ';
		}
		std::cout << std::endl;*/

		if (!std::equal(tsor.get_sizes().begin(), tsor.get_sizes().end(), expected_sizes.cbegin()))
		{
			throw std::runtime_error("TEST_3 in 'tensor_resize_testing_suit' failed!\n");
		}
		if (!std::equal(tsor.get_ranks().begin(), tsor.get_ranks().end(), expected_ranks.cbegin()))
		{
			throw std::runtime_error("TEST_3 in 'tensor_resize_testing_suit' failed!\n");
		}

		std::cout << "\tTEST 3 PASSED.\n";
	}

	void TEST_4()
	{
		tensor<int, 10> tsor(1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

		unsigned long long iterations;

		if constexpr (TENSORLIB_DEBUGGING)
		{
			iterations = 10;
		}
		else
			iterations = 100;

		for (unsigned long long i = 0; i < iterations; ++i) // check for memory leaks
		{
			tsor.resize(5, 5, 5, 5, 5, 5, 5, 5, 5, 5);
		}

		std::cout << "\tTEST 4 PASSED.\n";
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor resize tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();
		TEST_4();

		std::cout << "\n";
	}
}