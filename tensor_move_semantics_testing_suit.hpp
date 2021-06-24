#pragma once

#include <iostream>
#include <algorithm>
#include "helpers.hpp"
#include "tensor.hpp"

namespace tensor_move_semantics_testing_suit
{
	using namespace tensor_lib;

	void TEST_1()
	{
		tensor<int, 5> tsor(4u, 5u, 6u, 7u, 8u);

		tensor result (std::move(tsor));

		for (const auto& val : tsor.get_sizes())
		{
			if (val != 1)
				throw std::exception("TEST_1 in 'tensor_move_semantics_testing_suit' failed!\n");
		}

		for (size_t i = 0; i < 5; ++i)
		{
			if (result.get_sizes()[i] != i + 4)
				throw std::exception("TEST_1 in 'tensor_move_semantics_testing_suit' failed!\n");
		}

		std::cout << "\tTEST 1 PASSED.\n";
	}

	void TEST_2()
	{
		tensor<int, 5> tsor(4u, 5u, 6u, 7u, 8u), result;

		result = std::move(tsor);

		for (const auto& val : tsor.get_sizes())
		{
			if (val != 1)
				throw std::exception("TEST_2 in 'tensor_move_semantics_testing_suit' failed!\n");
		}

		for (size_t i = 0; i < 5; ++i)
		{
			if (result.get_sizes()[i] != i + 4)
				throw std::exception("TEST_2 in 'tensor_move_semantics_testing_suit' failed!\n");
		}

		std::cout << "\tTEST 2 PASSED.\n";
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor move semantics tests...\n\n";

		TEST_1();
		TEST_2();


		std::cout << "\n";
	}
}