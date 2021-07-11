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

		swap(tsor[0], tsor[1]); // specialization of swap()

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::exception("TEST_3 in 'tensor_iteration_testing_suit' failed!\n");

		std::cout << "\tTEST 3 PASSED.\n";
	}

	void TEST_4()
	{
		tensor<int, 3> tsor(2, 3, 4);
		tsor =
		{
			{
				{
					10, 28, 31, 33
				},
				{
					14, 11, 29, 32
				},// Sorting from here:
				{ //       [
					18, 15, 12, 30
				},
			},
			{
				{
					22, 19, 16, 13
				},// To here:
				{ //      ]
					25, 23, 20, 17
				},
				{
					277, 26, 24, 21
				},
			}
		};
		tensor<int, 3> expected(2, 3, 4);
		expected =
		{
			{
				{
					10, 28, 31, 33
				},
				{
					14, 11, 29, 32
				},
				{ 
					18, 15, 12, 13
				},
			},	//					 = > 12, 13, 16, 19, 22, 23, 25, 30
			{
				{
					16, 19, 22, 23
				},
				{
					25, 30, 20, 17
				},
				{
					277, 26, 24, 21
				},
			}
		};

		std::sort(tsor[0][2].begin() + 2, tsor[1][1].begin() + 2);

		if (!std::equal(tsor.cbegin(), tsor.cend(), expected.cbegin()))
			throw std::exception("TEST_4 in 'tensor_iteration_testing_suit' failed!\n");

		std::cout << "\tTEST 4 PASSED.\n";
	}

	void TEST_5()
	{
		tensor<int, 3> tsor_1(2, 3, 4);
		tsor_1 =
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

		tensor<float, 3> tsor_2(2, 3, 4);
		tsor_2 =
		{
			{
				{
					10.8f, 11.8f, 12.8f, 13.8f
				},
				{
					14.8f, 15.8f, 16.8f, 17.8f
				},
				{
					18.8f, 19.8f, 20.8f, 21.8f
				},
			},
			{
				{
					22.8f, 23.8f, 24.8f, 25.8f
				},
				{
					26.8f, 27.8f, 28.8f, 29.8f
				},
				{
					30.8f, 31.8f, 32.8f, 33.8f
				},
			}
		};

		tensor<int, 3> expected_1(2, 3, 4);
		expected_1 =
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

		tensor<float, 3> expected_2(2, 3, 4);
		expected_2 =
		{
			{
				{
					10.8f, 11.8f, 12.8f, 13.8f
				},
				{
					14.8f, 15.8f, 16.8f, 17.8f
				},
				{
					18.8f, 19.8f, 20.8f, 21.8f
				},
			},
			{
				{
					10.0f, 11.0f, 12.0f, 13.0f
				},
				{
					14.0f, 15.0f, 16.0f, 17.0f
				},
				{
					18.0f, 19.0f, 20.0f, 21.0f
				},
			}
		};

		swap(tsor_1[0], tsor_2[1]); // specialization of swap()

		if (!std::equal(tsor_1.cbegin(), tsor_1.cend(), expected_1.cbegin()))
			throw std::exception("TEST_5 in 'tensor_iteration_testing_suit' failed!\n");

		if (!std::equal(tsor_2.cbegin(), tsor_2.cend(), expected_2.cbegin()))
			throw std::exception("TEST_5 in 'tensor_iteration_testing_suit' failed!\n");

		std::cout << "\tTEST 5 PASSED.\n";
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor iteration tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();
		TEST_4();
		TEST_5();

		std::cout << "\n";
	}
}
