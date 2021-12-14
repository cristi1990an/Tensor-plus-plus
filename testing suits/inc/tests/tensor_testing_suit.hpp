#pragma once

#include "../../../inc/tensor.hpp"
#include "tensor_const_testing_suit.hpp"
#include "tensor_initialization_testing_suit.hpp"
#include "tensor_move_semantics_testing_suit.hpp"
#include "tensor_resize_testing_suit.hpp"
#include "tensor_iteration_testing_suit.hpp"
#include "tensor_replace_testing_suit.hpp"
#include "tensor_exceptions_testing_suit.hpp"

#include <iostream>

namespace tensor_testing_suit
{
	using namespace tensor_lib;

	void RUN_ALL_TESTS()
	{
		tensor_initialization_testing_suit::RUN_ALL();
		tensor_const_testing_suit::RUN_ALL();
		tensor_move_semantics_testing_suit::RUN_ALL();
		tensor_replace_testing_suit::RUN_ALL();
		tensor_resize_testing_suit::RUN_ALL();
		tensor_iteration_testing_suit::RUN_ALL();
		tensor_exceptions_testing_suit::RUN_ALL();
	}
}