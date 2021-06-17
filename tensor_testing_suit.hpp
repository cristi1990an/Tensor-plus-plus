#pragma once

#include <iostream>
#include "tensor.hpp"
#include "tensor_const_testing_suit.hpp"
#include "tensor_initialization_testing_suit.hpp"

namespace tensor_testing_suit
{
	using namespace tensor_lib;

	void RUN_ALL_TESTS()
	{
		tensor_initialization_testing_suit::RUN_ALL();
		tensor_const_testing_suit::RUN_ALL();
	}
}