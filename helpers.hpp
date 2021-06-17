#pragma once

#include <vector>
#include "tensor.hpp"

template<typename T, size_t Size>
void display(const std::array<T, Size>& arr)
{
    for (auto& val : arr)
        std::cout << val << " ";
    std::cout << std::endl;
}

template<typename T, size_t Size>
void display(const std::span<T>& arr)
{
    for (auto& val : arr)
        std::cout << val << " ";
    std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rank> requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
void display(const tensor_lib::const_subdimension<T, Rank>& mat)
{
    for (auto& val : mat)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

template<typename T, size_t Rank> requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
void display(const tensor_lib::const_subdimension<T, Rank>& mat)
{
    for (size_t i = 0; i < mat.order_of_current_dimension(); i++)
        display<T, Rank - 1>(mat[i]);
    std::cout << std::endl;
}

template<typename T, size_t Rank> requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
void display(const tensor_lib::const_subdimension<T, Rank>& mat)
{
    for (size_t i = 0; i < mat.order_of_current_dimension(); i++)
        display<T, Rank - 1>(mat[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rank> requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
void display(const tensor_lib::subdimension<T, Rank>& mat)
{
    for (auto& val : mat)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

template<typename T, size_t Rank> requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
void display(const tensor_lib::subdimension<T, Rank>& mat)
{
    for (size_t i = 0; i < mat.order_of_current_dimension(); i++)
        display<T, Rank - 1>(mat[i]);
    std::cout << std::endl;
}

template<typename T, size_t Rank> requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
void display(const tensor_lib::subdimension<T, Rank>& mat)
{
    for (size_t i = 0; i < mat.order_of_current_dimension(); i++)
        display<T, Rank - 1>(mat[i]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rank> requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
void display(const tensor_lib::tensor<T, Rank>& mat)
{
    for (auto& val : mat)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

template<typename T, size_t Rank> requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
void display(const tensor_lib::tensor<T, Rank>& mat)
{
    for (size_t i = 0; i < mat.order_of_current_dimension(); i++)
        display<T, Rank - 1>(mat[i]);
    std::cout << std::endl;
}

template<typename T, size_t Rank> requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
void display(const tensor_lib::tensor<T, Rank>& mat)
{
    
    for (size_t i = 0; i < mat.order_of_current_dimension(); i++)
        display<T, Rank - 1>(mat[i]);
        
}

template <typename T>
void display(const std::vector<T>& vec)
{
    for (const auto& value : vec)
        std::cout << value << " ";
    std::cout << std::endl;
}

template <typename Something>
void display(const std::vector<std::vector<Something>>& mat)
{
    for (const auto& vec : mat)
        display(vec);
}