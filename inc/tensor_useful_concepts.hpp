#pragma once

#include <concepts>
#include <iostream>
#include <type_traits>

namespace useful_concepts
{
	template <typename ... Args>
	concept integrals = (std::is_integral_v<std::remove_cv_t<Args>> && ...);

	template<typename T, typename ... Args>
	concept constructible_from_each = (std::is_constructible_v<T, Args> && ...);

	template <typename T, typename... U>
	concept same_as_common_type = requires ()
	{
		requires(std::is_same_v<typename std::common_type_t<U...>, T> == true);
	};

	template <typename T> 
	concept arithmetic = std::is_arithmetic_v<T>;

}