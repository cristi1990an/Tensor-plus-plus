#pragma once

#include "helpers.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <iostream>
#include <array>

namespace tensor_replace_testing_suit
{
	using namespace tensor_lib;

	void TEST_1()
	{
		tensor<int, 3> tsor(3, 2, 2);
		tensor<int, 3> result(3, 2, 2);
		std::array<int, 3 * 2 * 2> expected;

		int aux = 1;
		for (auto& val : tsor)
		{
			val = aux++;
		}

		aux = 1;
		for (auto& val : expected)
		{
			val = aux++;
		}

		std::fill(result.begin(), result.end(), 0);

		result.replace(tsor);

		if(!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_1 in 'tensor_replace_testing_suit' failed!\n");

		if (!std::equal(result.cbegin(), result.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_1 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 1 PASSED.\n";
	}

	void TEST_2()
	{
		tensor<int, 3> tsor(3, 2, 2);
		tensor<int, 2> result(2, 2);
		std::array<int, 2 * 2> expected;

		int aux = 1;
		for (auto& val : tsor)
		{
			val = aux++;
		}

		aux = 1;
		for (auto& val : expected)
		{
			val = aux++;
		}

		std::fill(result.begin(), result.end(), 0);

		result.replace(tsor[0]);

		if (!std::equal(tsor[0].cbegin(), tsor[0].cend(), expected.cbegin()))
			throw std::runtime_error("TEST_2 in 'tensor_replace_testing_suit' failed!\n");

		if (!std::equal(result.cbegin(), result.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_2 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 2 PASSED.\n";
	}

	void TEST_3()
	{
		tensor<int, 2> tsor(2, 2);
		tensor<int, 3> result(3, 2, 2);
		std::array<int, 2 * 2> expected;

		int aux = 1;
		for (auto& val : tsor)
		{
			val = aux++;
		}

		aux = 1;
		for (auto& val : expected)
		{
			val = aux++;
		}

		std::fill(result.begin(), result.end(), 0);

		result[0].replace(tsor);

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_3 in 'tensor_replace_testing_suit' failed!\n");

		if (!std::equal(result[0].cbegin(), result[0].cend(), expected.cbegin()))
			throw std::runtime_error("TEST_3 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 3 PASSED.\n";
	}

	void TEST_4()
	{
		tensor<int, 3> tsor(3, 2, 2);
		tensor<int, 3> result(3, 2, 2);
		std::array<int, 2 * 2> expected;

		int aux = 1;
		for (auto& val : tsor)
		{
			val = aux++;
		}

		aux = 1;
		for (auto& val : expected)
		{
			val = aux++;
		}

		std::fill(result.begin(), result.end(), 0);

		result[0].replace(tsor[0]);

		if (!std::equal(tsor[0].cbegin(), tsor[0].cend(), expected.cbegin()))
			throw std::runtime_error("TEST_4 in 'tensor_replace_testing_suit' failed!\n");

		if (!std::equal(result[0].cbegin(), result[0].cend(), expected.cbegin()))
			throw std::runtime_error("TEST_4 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 4 PASSED.\n";
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor replace tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();
		TEST_4();

		std::cout << "\n";
	}
}