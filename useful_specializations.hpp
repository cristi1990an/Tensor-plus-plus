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

	consteval std::size_t no_zero(const size_t val) noexcept
	{
		if (val)
			return val;
		return 1u;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename T>
	requires std::convertible_to<T, std::size_t>
		constexpr std::size_t multiply_all(const T val) noexcept
	{
		return val;
	}

	template<typename ... Sizes>
	requires useful_concepts::convertible_to_common_type<std::size_t, Sizes...>
		constexpr std::size_t multiply_all(const std::size_t first, const Sizes ... sizes) noexcept
	{
		return first * multiply_all(sizes...);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename T, size_t Size>
	consteval std::array<T, Size> value_initialize_array(T value)
	{
		std::array<T, Size> result{};
		std::fill(result.begin(), result.end(), value);
		return result;
	}
}
