#pragma once

#include "useful_concepts.hpp"

#include <type_traits>
#include <initializer_list>
#include <array>

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
		using type = std::initializer_list< typename nested_initializer_list_impl<T, N - 1>::type >;
	};

	template<typename T, size_t N>
	using nested_initializer_list = typename nested_initializer_list_impl<T, N>::type;

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename ... Args>
	constexpr bool contains_zero(const Args& ... values) noexcept
	{
		return !(... && values);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename T, size_t Size>
	consteval std::array<T, Size> value_initialize_array(T value)
	{
		std::array<T, Size> result{};
		std::fill(result.begin(), result.end(), value);
		return result;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	constexpr size_t exclude_zero(const size_t value) noexcept
	{
		if (value == 0)
			return 1;
		return value;
	}
}
