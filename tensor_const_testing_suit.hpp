#pragma once

#include <iostream>
#include "tensor.hpp"

namespace tensor_const_testing_suit
{
	using namespace tensor_lib;

	void TEST_1()
	{
		tensor<int, 3> tsor( 2, 2, 2 );

		[]([[maybe_unused]] const tensor<int, 3>& tsor)
		{
			if (! std::is_same_v<decltype(tsor[0]), const const_subdimension<int, 2>>)
			{
				throw std::runtime_error("TEST_1 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 1 PASSED.\n";
		}(tsor);
	}

	void TEST_2()
	{
		tensor<int, 3> tsor( 2, 2, 2 );

		[]([[maybe_unused]] tensor<int, 3>& tsor)
		{
			if (! std::is_same_v<decltype(tsor[0]), subdimension<int, 2>>)
			{
				throw std::runtime_error("TEST_2 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 2 PASSED.\n";
		}(tsor);
	}

	void TEST_3()
	{
		tensor<int, 3> tsor( 2, 2, 2 );

		[]([[maybe_unused]] const subdimension<int, 2>& subdim)
		{
			if (! std::is_same_v<decltype(subdim[0]), const const_subdimension<int, 1>>)
			{
				throw std::runtime_error("TEST_3 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 3 PASSED.\n";
		}(tsor[0]);
	}

	void TEST_4()
	{
		tensor<int, 3> tsor( 2, 2, 2 );
		auto subdim = tsor[0];

		[]([[maybe_unused]] subdimension<int, 2>& subdim)
		{
			if (! std::is_same_v<decltype(subdim[0]), subdimension<int, 1>>)
			{
				throw std::runtime_error("TEST_4 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 4 PASSED.\n";
		}(subdim);
	}

	void TEST_5()
	{
		tensor<int, 3> tsor( 2, 2, 2 );
		const_subdimension<int, 3> subdim(tsor);

		[]([[maybe_unused]] const const_subdimension<int, 3>& subdim)
		{
			if (! std::is_same_v<decltype(subdim[0]), const const_subdimension<int, 2>>)
			{
				throw std::runtime_error("TEST_5 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 5 PASSED.\n";
		}(subdim);
	}

	void TEST_6()
	{
		tensor<int, 3> tsor( 2, 2, 2 );
		
		[]([[maybe_unused]] const tensor<int, 3>& tsor)
		{
			if (! std::is_same_v<decltype(tsor[0][0][0]), const int &>)
			{
				throw std::runtime_error("TEST_6 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 6 PASSED.\n";
		}(tsor);
	}

	void TEST_7()
	{
		tensor<int, 3> tsor( 2, 2, 2 );
		subdimension<int, 3> subdim(tsor);

		[]([[maybe_unused]] const subdimension<int, 3>& subdim)
		{
			if (! std::is_same_v<decltype(subdim[0][0][0]), const int&>)
			{
				throw std::runtime_error("TEST_7 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 7 PASSED.\n";
		}(subdim);
	}

	void TEST_8()
	{
		tensor<int, 3> tsor(2, 2, 2);
		const_subdimension<int, 3> subdim(tsor);

		[]([[maybe_unused]] const const_subdimension<int, 3>& subdim)
		{
			if (! std::is_same_v<decltype(subdim[0][0][0]), const int&>)
			{
				throw std::runtime_error("TEST_8 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 8 PASSED.\n";
		}(subdim);
	}

	void TEST_9()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] tensor<int, 3>& tsor)
		{
			if (! std::is_same_v<decltype(tsor[0][0][0]), int&>)
			{
				throw std::runtime_error("TEST_9 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 9 PASSED.\n";
		}(tsor);
	}

	void TEST_10()
	{
		tensor<int, 3> tsor(2, 2, 2);
		subdimension<int, 3> subdim(tsor);

		[]([[maybe_unused]] subdimension<int, 3>& subdim)
		{
			if (! std::is_same_v<decltype(subdim[0][0][0]), int&>)
			{
				throw std::runtime_error("TEST_10 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 10 PASSED.\n";
		}(subdim);
	}

	void TEST_11()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] tensor<int, 3>& tsor)
		{
			auto it = tsor.begin();

			if (! std::is_same_v<decltype(it), _tensor_common<int>::iterator>)
			{
				throw std::runtime_error("TEST_11 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 11 PASSED.\n";
		}(tsor);
	}

	void TEST_12()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] const tensor<int, 3>& tsor)
		{
			auto it = tsor.begin();

			if (not std::is_same_v<decltype(it), _tensor_common<int>::const_iterator>)
			{
				throw std::runtime_error("TEST_12 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 12 PASSED.\n";
		}(tsor);
	}

	void TEST_13()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] tensor<int, 3>& tsor)
		{
			auto it = tsor[0].begin();

			if (! std::is_same_v<decltype(it), _tensor_common<int>::iterator>)
			{
				throw std::runtime_error("TEST_13 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 13 PASSED.\n";
		}(tsor);
	}

	void TEST_14()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] const tensor<int, 3>& tsor)
		{
			auto it = tsor[0].begin();

			if (! std::is_same_v<decltype(it), _tensor_common<int>::const_iterator>)
			{
				throw std::runtime_error("TEST_14 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 14 PASSED.\n";
		}(tsor);
	}

	void TEST_15()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] tensor<int, 3>& tsor)
		{
			auto it = tsor.begin();

			if (! std::is_same_v<decltype(it), _tensor_common<int>::iterator>)
			{
				throw std::runtime_error("TEST_15 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 15 PASSED.\n";
		}(tsor);
	}

	void TEST_16()
	{
		tensor<int, 3> tsor(2, 2, 2);

		[]([[maybe_unused]] tensor<int, 3>& tsor)
		{
			auto it = tsor.cbegin();

			if (! std::is_same_v<decltype(it), _tensor_common<int>::const_iterator>)
			{
				throw std::runtime_error("TEST_16 in 'tensor_const_testing_suit' failed!\n");
			}
			else
				std::cout << "\tTEST 16 PASSED.\n";
		}(tsor);
	}

	void RUN_ALL()
	{
		std::cout << "Running tensor const correctness tests...\n\n";

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
		TEST_16();

		std::cout << "\n";
	}
}