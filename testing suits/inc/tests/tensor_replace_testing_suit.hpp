#pragma once

#include "../../../inc/tensor.hpp"

#include <algorithm>
#include <iostream>
#include <array>
#include <list>

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

	void TEST_5()
	{
		tensor<int, 3> tsor(3, 2, 2);
		const_subdimension<int, 3> source(tsor);
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

		result.replace(source);

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_5 in 'tensor_replace_testing_suit' failed!\n");

		if (!std::equal(result.cbegin(), result.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_5 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 5 PASSED.\n";
	}

	void TEST_6()
	{
		tensor<int, 3> tsor(3, 2, 2);
		const_subdimension<int, 3> source(tsor);
		tensor<int, 3> result(3, 2, 2);
		subdimension<int, 3> destination(result);
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

		destination.replace(source);

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_6 in 'tensor_replace_testing_suit' failed!\n");

		if (!std::equal(result.cbegin(), result.cend(), expected.cbegin()))
			throw std::runtime_error("TEST_6 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 6 PASSED.\n";
	}

	void TEST_7()
	{
		tensor<int, 3> tsor(3, 2, 2); // 12 in size
		std::list values = { 1,2,3,4,5,6,7,8,9,10,11,12 };

		tsor.replace(values.cbegin(), values.cend());

		for (int i = 1; const auto val : tsor)
		{
			if (val != i++)
			{
				throw std::runtime_error("TEST_7 in 'tensor_replace_testing_suit' failed!\n");
			}
		}

		std::cout << "\tTEST 7 PASSED.\n";
	}

	void TEST_8()
	{
		tensor<int, 3> tsor(3, 2, 2); // 12 in size
		std::list values = { 1,2,3,4 };

		tsor[0].replace(values.cbegin(), values.cend());
		tsor[1].replace(values.cbegin(), values.cend());
		tsor[2].replace(values.cbegin(), values.cend());

		auto expected = { 1,2,3,4,1,2,3,4,1,2,3,4 };

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.begin(), expected.end()))
			throw std::runtime_error("TEST_8 in 'tensor_replace_testing_suit' failed!\n");

		std::cout << "\tTEST 8 PASSED.\n";
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor replace tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();
		TEST_4();
		TEST_5();
		TEST_6();
		TEST_7();
		TEST_8();

		std::cout << "\n";
	}
}