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
#include <type_traits>
#include <iterator>

namespace tensor_testing_suit
{
	using namespace tensor_lib;

	#define ASSERT_SAME_AS(T1, T2) static_assert(std::same_as<T1, T2>)
	#define METHOD_RESULT_TYPE(CLASS_TYPE, METHOD) decltype(std::declval<CLASS_TYPE>().METHOD())
	#define DEREFERENCE_TYPE(ITERATOR) std::remove_reference_t<decltype(*std::declval<ITERATOR>())>

	template<typename Iterator, typename UnderlyingType>
	consteval void ITERATOR_ASSERTS() noexcept
	{
		using traits = std::iterator_traits<Iterator>;
		constexpr bool IS_CONST_ITERATOR = std::is_const_v<typename traits::value_type>;

		static_assert(std::is_const_v <UnderlyingType> == IS_CONST_ITERATOR);

		ASSERT_SAME_AS(typename traits::difference_type, std::ptrdiff_t);
		ASSERT_SAME_AS(typename traits::iterator_category, std::contiguous_iterator_tag);
		ASSERT_SAME_AS(typename traits::pointer, UnderlyingType*);
		ASSERT_SAME_AS(typename traits::reference, UnderlyingType&);
		ASSERT_SAME_AS(typename traits::value_type, UnderlyingType);

		static_assert(std::input_iterator<Iterator>);
		static_assert(std::random_access_iterator<Iterator>);
		static_assert(std::forward_iterator<Iterator>);

		if constexpr (!IS_CONST_ITERATOR)
		{
			static_assert(std::input_or_output_iterator<Iterator>);
			static_assert(std::input_or_output_iterator<Iterator>);
		}
	}

	template<typename Container, bool ForceConst = false>
	consteval void BEGIN_END_ASSERTS() noexcept
	{
		constexpr bool IS_CONST_CONTAINER = std::is_const_v<Container> || ForceConst;

		using begin_result_t = METHOD_RESULT_TYPE(Container, begin);
		using cbegin_result_t = METHOD_RESULT_TYPE(Container, cbegin);
		using end_result_t = METHOD_RESULT_TYPE(Container, end);
		using cend_result_t = METHOD_RESULT_TYPE(Container, cend);

		constexpr bool begin_value_t_is_const = std::is_const_v<DEREFERENCE_TYPE(begin_result_t)>;
		constexpr bool cbegin_value_t_is_const = std::is_const_v<DEREFERENCE_TYPE(cbegin_result_t)>;
		constexpr bool end_value_t_is_const = std::is_const_v<DEREFERENCE_TYPE(end_result_t)>;
		constexpr bool cend_value_t_is_const = std::is_const_v<DEREFERENCE_TYPE(cend_result_t)>;

		static_assert(cbegin_value_t_is_const && cend_value_t_is_const);

		if constexpr (IS_CONST_CONTAINER)
		{
			static_assert(begin_value_t_is_const && end_value_t_is_const);
		}
		else
		{
			static_assert(!begin_value_t_is_const && !end_value_t_is_const);
		}

		ITERATOR_ASSERTS<begin_result_t, DEREFERENCE_TYPE(begin_result_t)>();
		ITERATOR_ASSERTS<cbegin_result_t, DEREFERENCE_TYPE(cbegin_result_t)>();
		ITERATOR_ASSERTS<end_result_t, DEREFERENCE_TYPE(end_result_t)>();
		ITERATOR_ASSERTS<cend_result_t, DEREFERENCE_TYPE(cend_result_t)>();
	}

	void RUN_ALL_TESTS()
	{
		tensor_initialization_testing_suit::RUN_ALL();
		tensor_const_testing_suit::RUN_ALL();
		tensor_move_semantics_testing_suit::RUN_ALL();
		tensor_replace_testing_suit::RUN_ALL();
		tensor_resize_testing_suit::RUN_ALL();
		tensor_iteration_testing_suit::RUN_ALL();
		tensor_exceptions_testing_suit::RUN_ALL();

		BEGIN_END_ASSERTS<tensor_lib::tensor<int, 10>>();
		BEGIN_END_ASSERTS<tensor_lib::subdimension<int, 10>>();
		BEGIN_END_ASSERTS<tensor_lib::const_subdimension<int, 10>, true>();
		BEGIN_END_ASSERTS<const tensor_lib::tensor<int, 10>>();
		BEGIN_END_ASSERTS<const tensor_lib::subdimension<int, 10>>();
		BEGIN_END_ASSERTS<const tensor_lib::const_subdimension<int, 10>>();
	}
}