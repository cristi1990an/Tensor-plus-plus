#pragma once

#include "helpers.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace tensor_initialization_testing_suit
{
	using namespace tensor_lib;

#define nested_initializer_list {{{{ 10, 11 },{ 12, 13 },{ 14, 15 },},{ { 16, 17 }, { 18, 19 }, { 20, 21 }, }, { { 22, 23 }, { 24, 25 }, { 26, 27 }, }, { { 28, 29 }, { 30, 31 }, { 32, 33 }, } }, { { { 34, 35 }, { 36, 37 }, { 38, 39 }, }, { { 40, 41 }, { 42, 43 }, { 44, 45 }, }, { { 46, 47 }, { 48, 49 }, { 50, 51 }, }, { { 52, 53 }, { 54, 55 }, { 56, 57 }, } }, { { { 58, 59 }, { 60, 61 }, { 62, 63 }, }, { { 64, 65 }, { 66, 67 }, { 68, 69 }, }, { { 70, 71 }, { 72, 73 }, { 74, 75 }, }, { { 76, 77 }, { 78, 79 }, { 80, 81 }, } }, { { { 82, 83 }, { 84, 85 }, { 86, 87 }, }, { { 88, 89 }, { 90, 91 }, { 92, 93 }, }, { { 94, 95 }, { 96, 97 }, { 98, 99 }, }, { { 10, 11 }, { 12, 13 }, { 14, 15 }, } }, { { { 16, 17 }, { 18, 19 }, { 20, 21 }, }, { { 22, 23 }, { 24, 25 }, { 26, 27 }, }, { { 28, 29 }, { 30, 31 }, { 32, 33 }, }, { { 34, 35 }, { 36, 37 }, { 38, 39 }, } } }

	void TEST_1()
	{
		tensor<int, 4> tsor_1(5, 4, 3, 2), tsor_2(5, 4, 3, 2);
		int val = 10;

		tsor_1 = nested_initializer_list;

		for (auto& value : tsor_2)
		{
			if (val == 100)
				val = 10;
			value = val++;
		}

		if (not std::equal(tsor_1.cbegin(), tsor_1.cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_1 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 1 PASSED.\n";
	}

	void TEST_2()
	{
		tensor<int, 2> tsor_1(2, 20), tsor_2(2, 20);

		std::fill(tsor_1.begin(), tsor_1.end(), 5);
		std::fill(tsor_1[1].begin(), tsor_1[1].end(), 7);

		for (size_t i = 0; i < tsor_2[0].order_of_current_dimension(); i++)
		{
			tsor_2[0][i] = 5;
		}

		for (size_t i = 0; i < tsor_2[1].order_of_current_dimension(); i++)
		{
			tsor_2[1][i] = 7;
		}

		if (not std::equal(tsor_1.cbegin(), tsor_1.cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_2 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 2 PASSED.\n";
	}

	void TEST_3()
	{
		tensor<int, 3> tsor_1(2, 2, 2), tsor_2(2, 2, 2);
		int val = 0;

		for (auto& value : tsor_1)
		{
			value = val++;
		}

		tsor_2 = { 0, 1, 2, 3, 4, 5, 6, 7 };

		if (not std::equal(tsor_1.cbegin(), tsor_1.cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_3 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 3 PASSED.\n";
	}

	void TEST_4()
	{
		tensor<int, 3> tsor_1(2, 2, 2), tsor_2(2, 2, 2);

		tsor_1[0][0] = { 0, 0 };
		tsor_1[0][1] = { 0, 0 };
		tsor_1[1][0] = { 4, 5 };
		tsor_1[1][1] = { 6, 7 };

		tsor_2[0] = { 0, 0, 0, 0 };
		tsor_2[1] = { 4, 5, 6, 7 };

		if (not std::equal(tsor_1.cbegin(), tsor_1.cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_4 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 4 PASSED.\n";
	}

	void TEST_5()
	{
		tensor<int, 3> tsor_1(2, 2, 2), tsor_2(2, 2, 2);
		subdimension sub_1(tsor_1), sub_2(tsor_2);

		sub_1 = { 1, 2, 3, 4, 5, 6, 7, 8 };
		sub_2 =
		{
			{
				{ 1, 2 },
				{ 3, 4 }
			},
			{
				{ 5, 6 },
				{ 7, 8 }
			}
		};

		if (not std::equal(tsor_1.cbegin(), tsor_1.cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_5 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 5 PASSED.\n";
	}

	void TEST_6()
	{
		tensor<int, 2> tsor_1(2, 5);
		subdimension<int, 1> sub_1(tsor_1[1]);

		tensor<int, 1> tsor_2(5);

		tsor_2 = { 1, 2, 3, 4, 5 };

		std::copy(tsor_2.cbegin(), tsor_2.cend(), tsor_1[0].begin());
		std::copy(tsor_2.cbegin(), tsor_2.cend(), sub_1.begin());

		if (not std::equal(tsor_1[0].cbegin(), tsor_1[0].cend(), tsor_2.cbegin()) or not std::equal(tsor_1[1].cbegin(), tsor_1[1].cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_6 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 6 PASSED.\n";
	}

	void TEST_7()
	{
		tensor<int, 2> tsor(2, 5);

		tsor = { {1, 2, 3, 4, 5}, {7, 7, 7, 7, 7} };

		tensor real_copy(tsor[0]);


		subdimension reference(tsor[0]);

		if (not std::equal(real_copy.cbegin(), real_copy.cend(), reference.cbegin()))
		{
			throw std::runtime_error("TEST_7 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 7 PASSED.\n";
	}

	void TEST_8()
	{
		tensor<int, 3> tsor_1(10, 10, 10), tsor_2(10, 10, 10);
		int val = 0;

		for (size_t i = 0; i < tsor_1.order_of_dimension(0); i++)
			for (size_t j = 0; j < tsor_1.order_of_dimension(1); j++)
				for (size_t k = 0; k < tsor_1.order_of_dimension(2); k++)
					tsor_1[i][j][k] = val++;
		val = 0;

		for (auto& value : tsor_2)
		{
			value = val++;
		}

		if (not std::equal(tsor_1.cbegin(), tsor_1.cend(), tsor_2.cbegin()))
		{
			throw std::runtime_error("TEST_8 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 8 PASSED.\n";
	}

	void TEST_9()
	{
		tensor<int, 5> tsor(3, 3, 3, 3, 3);

		std::fill(tsor.begin(), tsor.end(), 3);

		if (std::pow(3, 6) != std::accumulate(tsor.cbegin(), tsor.cend(), 0))
		{
			throw std::runtime_error("TEST_9 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 9 PASSED.\n";
	}

	void TEST_10()
	{
		tensor<int, 5> tsor(1, 2, 3, 4, 5);

		auto sizes = tsor.get_ranks();
		std::array<size_t, 5> expected = { 1, 2, 3, 4, 5 };

		if (!std::equal(sizes.begin(), sizes.end(), expected.begin()))
		{
			for (const auto& val : sizes)
			{
				std::cout << val << ' ';
			}
			throw std::runtime_error("TEST_10 in 'tensor_initialization_testing_suit' failed!\n");
		}
		else
			std::cout << "\tTEST 10 PASSED.\n";
	}

	void TEST_11()
	{
		std::initializer_list<std::initializer_list<std::initializer_list<int>>> nested_init_list =
		{
			{
				{1, 2, 3}, {1, 2, 3}, {1, 2, 3},
			},
			{
				{1, 2, 3}, {1, 2, 3}, {1, 2, 3}
			},
			{
				{1, 2, 3}, {1, 2, 3}, {1, 2, 3},
			},
			{
				{1, 2, 3}, {1, 2, 3}, {1, 2, 3}
			}
		};

		tensor<int, 3> three_dim_tensor = nested_init_list;
		tensor<int, 3> explicit_dim_tensor(4, 3, 3);
		explicit_dim_tensor = nested_init_list;

		if (!std::equal(three_dim_tensor.cbegin(), three_dim_tensor.cend(), explicit_dim_tensor.cbegin()))
		{
			throw std::runtime_error("TEST_11 in 'tensor_initialization_testing_suit' failed!\n");
		}

		std::cout << "\tTEST 11 PASSED.\n";
	}

	struct Helper_Class_1
	{
		std::string _string{};

		Helper_Class_1()
		{
			throw std::runtime_error("Default constructor called!\n");
		}
		template<typename T>
		Helper_Class_1(T&& value)
			: _string(std::forward<T>(value))
		{

		}
	};

	void TEST_12()
	{
		tensor<Helper_Class_1, 3> tsor(1, 2, 3, "Cristi");

		for (const auto& obj : tsor)
		{
			if (obj._string != "Cristi")
				throw std::runtime_error("TEST_12 in 'tensor_initialization_testing_suit' failed!\n");
		}

		if(tsor.size_of_current_tensor() != 6)
			throw std::runtime_error("TEST_12 in 'tensor_initialization_testing_suit' failed!\n");

		std::cout << "\tTEST 12 PASSED.\n";
	}


	struct Helper_Class_2
	{
		int _int{};
		float _float{};
		std::string _string{};

		Helper_Class_2()
		{
			throw std::runtime_error("Default constructor called!\n");
		}
		template<typename T>
		Helper_Class_2(const int integer, const float fp, T&& value)
			: _int(integer)
			, _float(fp)
			, _string(std::forward<T>(value))
		{

		}
	};

	void TEST_13()
	{
		tensor<Helper_Class_2, 3> tsor(1, 2, 3, 5, 5.5f, "Cristi");

		for (const auto& obj : tsor)
		{
			if (obj._int != 5 || obj._float != 5.5f || obj._string != "Cristi")
				throw std::runtime_error("TEST_13 in 'tensor_initialization_testing_suit' failed!\n");
		}

		if (tsor.size_of_current_tensor() != 6)
			throw std::runtime_error("TEST_13 in 'tensor_initialization_testing_suit' failed!\n");


		std::cout << "\tTEST 13 PASSED.\n";
	}


	struct Helper_Class_3
	{
		int value = 5;
		static size_t counter;

		Helper_Class_3()
		{

		}
		Helper_Class_3(const Helper_Class_3&)
		{
			counter++;
			if (counter > 1 * 2 * 3)
			{
				throw std::runtime_error("Object was copied!\n");
			}
		}
	};
	size_t Helper_Class_3::counter = 0;

	struct Helper_Class_4
	{
		Helper_Class_3 obj;

		Helper_Class_4(const Helper_Class_3& other)
			: obj(other)
		{
			
		}
	};

	void TEST_14()
	{
		tensor<Helper_Class_4, 3> tsor(1, 2, 3, Helper_Class_3());

		for (const auto& object : tsor)
		{
			if (object.obj.value != 5)
				throw std::runtime_error("TEST_14 in 'tensor_initialization_testing_suit' failed!\n");
		}

		if (tsor.size_of_current_tensor() != 6)
			throw std::runtime_error("TEST_14 in 'tensor_initialization_testing_suit' failed!\n");


		std::cout << "\tTEST 14 PASSED.\n";
	}

	void TEST_15()
	{
		tensor<std::string, 4> tsor(5, 4, 3, 2, "Some long string I'm writing out of the top of my head...");

		for (size_t index = 0; index != tsor.size_of_current_tensor(); index++)
		{
			if (*(tsor.cbegin() + index) != "Some long string I'm writing out of the top of my head...")
				throw std::runtime_error("TEST_15 in 'tensor_initialization_testing_suit' failed!\n");
		}

		if (tsor.size_of_current_tensor() != 120)
			throw std::runtime_error("TEST_15 in 'tensor_initialization_testing_suit' failed!\n");


		std::cout << "\tTEST 15 PASSED.\n";
	}


	void RUN_ALL()
	{
		std::cout << "Running tensor initialization tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();
		TEST_4();
		TEST_5();
		TEST_6();
		TEST_7();
		TEST_8();
		TEST_9();
		TEST_10();
		TEST_11();
		TEST_12();
		TEST_13();
		TEST_14();
		TEST_15();

		std::cout << "\n";
	}
}