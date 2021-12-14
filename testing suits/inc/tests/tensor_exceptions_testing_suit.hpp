#pragma once

#include "../../../inc/tensor.hpp"

#include <iostream>

namespace tensor_exceptions_testing_suit
{
	using namespace tensor_lib;

	void TEST_1()
	{
		tensor<int, 3> tsor{ 2, 2, 2 };

		try
		{
			tsor = { 1, 2, 3 }; // initializer list should have 8 elements
		}
		catch (...)
		{
			std::cout << "\tTEST 1 PASSED.\n";
			return;
		}

		throw std::runtime_error("TEST_1 in 'tensor_exceptions_testing_suit' failed!\n");
	}

	void TEST_2()
	{
		try
		{
			tensor<int, 3> tsor
			{
				{
					{ 1, 2, 3},
					{ 1, 2, 3, 4} // initializer list with one element too many compared to the rest
				},

				{
					{ 1, 2, 3},
					{ 1, 2, 3}
				}
			};
		}
		catch (...)
		{
			std::cout << "\tTEST 2 PASSED.\n";
			return;
		}

		throw std::runtime_error("TEST_2 in 'tensor_exceptions_testing_suit' failed!\n");
	}

	void TEST_3()
	{
		try
		{
			tensor<int, 3> tsor
			{
				{
					{ 1, 2, 3},
					{ 1, 2, 3} 
				},

				{
					{ 1, 2, 3},
					{ 1, 2, 3},
					{ 1, 2, 3} // initializer list with one nested initializer list too many
				}
			};
		}
		catch (...)
		{
			std::cout << "\tTEST 3 PASSED.\n";
			return;
		}

		throw std::runtime_error("TEST_3 in 'tensor_exceptions_testing_suit' failed!\n");
	}

	void TEST_4()
	{
		try
		{
			tensor<int, 3> tsor;
			tsor =
			{
				{
					{ 1, 2, 3},
					{ 1, 2, 3, 4} // initializer list with one element too many compared to the rest
				},

				{
					{ 1, 2, 3},
					{ 1, 2, 3}
				}
			};
		}
		catch (...)
		{
			std::cout << "\tTEST 4 PASSED.\n";
			return;
		}

		throw std::runtime_error("TEST_4 in 'tensor_exceptions_testing_suit' failed!\n");
	}

	void TEST_5()
	{
		try
		{
			tensor<int, 3> tsor;
			tsor =
			{
				{
					{ 1, 2, 3},
					{ 1, 2, 3}
				},

				{
					{ 1, 2, 3},
					{ 1, 2, 3},
					{ 1, 2, 3} // initializer list with one nested initializer list too many
				}
			};
		}
		catch (...)
		{
			std::cout << "\tTEST 5 PASSED.\n";
			return;
		}

		throw std::runtime_error("TEST_5 in 'tensor_exceptions_testing_suit' failed!\n");
	}

	void TEST_xx()
	{
		tensor<int, 2> t1(2, 2), t2(2, 2), t3(2, 1);
		t1 = { 1,2,3,4 };
		t2 = { 5,6,7,8 };
		t3 = { 9, 10 }; // only 2 elements

		tensor<int, 3> combined(4, 2, 2);

		try
		{
			combined.replace(t1, t2, t3);
		}
		catch (...)
		{
			std::cout << "\tTEST 29 PASSED.\n";
			return;
		}

		throw std::runtime_error("TEST_29 in 'tensor_exceptions_testing_suit' failed!\n");
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor exceptions tests...\n\n";

		TEST_1();
		TEST_2();
		TEST_3();
		TEST_4();
		TEST_5();
		TEST_xx();

		std::cout << "\n";
	}
}