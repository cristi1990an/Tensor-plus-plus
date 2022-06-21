#pragma once

#include "tensor_useful_concepts.hpp"

#include <type_traits>
#include <initializer_list>

namespace useful_specializations
{
	// "nested_initializer_list<int, 3>" expands into "initializer_list<initializer_list<initializer_list<int>>>"

	template<typename, size_t>
	struct nested_initializer_list_impl;

	template<typename T>
	struct nested_initializer_list_impl<T, 1>
	{
		using type = std::initializer_list<T>;
	};

	template<typename T, size_t N>
	struct nested_initializer_list_impl
	{
		using type = std::initializer_list<typename nested_initializer_list_impl<T, N - 1>::type >;
	};

	template<typename T, size_t N>
	using nested_initializer_list_t = typename nested_initializer_list_impl<T, N>::type;
}
