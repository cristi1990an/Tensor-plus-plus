#pragma once

#include <type_traits>
#include <concepts>
#include <iostream>

namespace useful_concepts
{
    template <typename T>
    concept addable =
        requires (T x, T y)
    {
        x + y;
    };

    template <typename T, typename U>
    concept addable_to =
        requires (T t, U u)
    {
        t + u;
    };

    template <typename T, typename U>
    concept mutually_addable_to_each_other =
        addable_to<T, U> &&
        addable_to<U, T>;

    template <typename T>
    concept substractable =
        requires (T x, T y)
    {
        x - y;
    };

    template <typename T, typename U>
    concept substractable_from =
        requires (T t, U u)
    {
        u - t;
    };

    template <typename T, typename U>
    concept mutually_substractable_from_each_other =
        substractable_from<T, U> &&
        substractable_from<U, T>;

    template <typename T>
    concept multipliable =
        requires (T x, T y)
    {
        x* y;
    };

    template <typename T, typename U>
    concept multipliable_with =
        requires (T t, U u)
    {
        t* u;
    };

    template <typename T, typename U>
    concept mutually_multipliable_with_each_other =
        multipliable_with<T, U> &&
        multipliable_with<U, T>;

    template <typename T>
    concept divisible =
        requires (T x, T y)
    {
        x / y;
    };

    template <typename T, typename U>
    concept divisible_with = requires (T t, U u)
    {
        t / u;
    };

    template <typename T, typename U>
    concept mutually_divisible_with_each_other =
        divisible_with<T, U> &&
        divisible_with<U, T>;

    template <typename... T, size_t SUM>
    concept size_of_parameter_pack_equals = requires ()
    {
        requires(sizeof...(T) == SUM);
    };

    template <typename T, size_t VAL>
    concept size_equals = requires ()
    {
        requires(sizeof(T) == VAL);
    };

    template <typename T, typename U>
    concept constructable_from = requires()
    {
        requires(std::is_constructible_v<T, U> == true);
    };

    template <typename T, typename... U>
    concept constructable_from_common_type = requires ()
    {
        requires(std::is_constructible_v<T, typename std::common_type_t<U...>> == true);
    };

    template <typename T, typename... U>
    concept convertible_to_common_type =
        std::convertible_to<typename std::common_type<U...>::type, T>;

    template <typename T, typename... U>
    concept same_as_common_type = requires ()
    {
        requires(std::is_same_v<typename std::common_type_t<U...>, T> == true);
    };

    template <typename T, typename... U>
    concept not_same_as_common_type = requires ()
    {
        requires(std::is_same_v<typename std::common_type_t<U...>, T> == false);
    };

    template <typename T, typename U>
    concept mutually_convertible_from_each_other =
        std::convertible_to<T, U> &&
        std::convertible_to<U, T>;

    template <typename T>
    concept ostream_compatible =
        requires (T value)
    {
        std::cout << value;
    };

    template <typename T>
    concept signed_type =
        requires(T t)
    {
        requires(std::is_signed<T>().value == true);
    };

    template <typename T>
    concept is_invertible =
        signed_type<T> &&
        requires (T t)
    {
        t = -t;
    };

    template <typename T>
    concept has_zero =
        requires(T t)
    {
        t = static_cast<T>(0);
    };

    template <typename T>
    concept default_initialized_with_zero =
        has_zero<T> &&
        std::default_initializable<T> &&
        requires ()
    {
        requires(static_cast<T>(0) == T{});
    };

    template <typename T>
    concept general_arithmetics =
        addable<T> &&
        substractable<T> &&
        multipliable<T> &&
        divisible<T> &&
        std::regular<T> &&
        signed_type<T> &&
        default_initialized_with_zero<T>;


    template <typename T, typename U, T t, U u>
    concept is_greater_than =
        std::three_way_comparable_with<T, U> &&
        requires()
    {
        requires(t > u);
    };

    template <typename T, typename U, T t, U u>
    concept is_equal_to =
        std::equality_comparable_with<T, U> &&
        requires()
    {
        requires(t == u);
    };

    template <typename T, typename U, T t, U u>
    concept is_equal_or_greater_than =
        std::three_way_comparable_with<T, U> &&
        requires()
    {
        requires(t >= u);
    };

    template <typename T, typename U, T t, U u>
    concept is_smaller_than =
        std::three_way_comparable_with<T, U> &&
        requires()
    {
        requires(t < u);
    };

    template <typename T, typename U, T t, U u>
    concept is_equal_or_smaller_than =
        std::three_way_comparable_with<T, U> &&
        requires()
    {
        requires(t <= u);
    };

    template <typename T, typename U>
    concept can_static_cast_to = requires (T t, U u)
    {
        u = static_cast<U>(t);
    };

    template <typename T, typename U>
    concept is_same = requires()
    {
        requires(std::is_same_v<T, U> == true);
    };

}