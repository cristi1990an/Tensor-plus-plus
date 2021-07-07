#pragma once

#include <iostream>
#include <algorithm>
#include <array>
#include "helpers.hpp"
#include "tensor.hpp"

namespace tensor_iteration_testing_suit
{
	using namespace tensor_lib;

	void TEST_1()
	{
		tensor<int, 2> tsor(2, 4);
		tsor = 
		{
			{
				1, 2, 3, 4
			},
			{
				5, 6, 7, 8
			}
		};

		std::array<int, 6> result;
		std::array expected = { 3,4,5,6,7,8 };

		std::copy(tsor[0].cbegin() + 2, tsor.cend(), result.begin());

		if(!std::equal(result.cbegin(), result.cend(), expected.cbegin()))
			throw std::exception("TEST_1 in 'tensor_iteration_testing_suit' failed!\n");

		std::cout << "\tTEST 1 PASSED.\n";
	}

	void TEST_2()
	{
		tensor<int, 3> tsor(2, 3, 4);
		tsor =
		{
			{
				{
					10, 11, 12, 13
				},
				{
					14, 15, 16, 17
				},
				{
					18, 19, 20, 21
				},
			},
			{
				{
					22, 23, 24, 25
				},
				{
					26, 27, 28, 29
				},
				{
					30, 31, 32, 33
				},
			}
		};
		tensor<int, 3> expected(2, 3, 4);
		expected = 
		{
			{
				{
					10, 11, 12, 13
				},
				{
					14, 15,  0,  0
				},
				{
					 0,  0,  0,  0
				},
			},
			{
				{
					22, 23, 24, 25
				},
				{
					 0,  0,  0,  0
				},
				{
					 0, 31, 32, 33
				},
			}
		};

		std::fill(tsor[0][1].begin() + 2,	tsor[0].end(),			0);
		std::fill(tsor[1][1].begin(),		tsor[1][2].begin() + 1, 0);

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::exception("TEST_2 in 'tensor_iteration_testing_suit' failed!\n");

		std::cout << "\tTEST 2 PASSED.\n";
	}

	void TEST_3()
	{
		tensor<int, 3> tsor(2, 3, 4);
		tsor =
		{
			{
				{
					10, 11, 12, 13
				},
				{
					14, 15, 16, 17
				},
				{
					18, 19, 20, 21
				},
			},
			{
				{
					22, 23, 24, 25
				},
				{
					26, 27, 28, 29
				},
				{
					30, 31, 32, 33
				},
			}
		};
		tensor<int, 3> expected(2, 3, 4);
		expected =
		{
			{
				{
					22, 23, 24, 25
				},
				{
					26, 27, 28, 29
				},
				{
					30, 31, 32, 33
				},
			},
			{
				{
					10, 11, 12, 13
				},
				{
					14, 15, 16, 17
				},
				{
					18, 19, 20, 21
				},
			}
		};

		swap(tsor[0], tsor[1]);

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::exception("TEST_3 in 'tensor_iteration_testing_suit' failed!\n");

		std::cout << "\tTEST 3 PASSED.\n";
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor iteration tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();


		std::cout << "\n";
	}
}
