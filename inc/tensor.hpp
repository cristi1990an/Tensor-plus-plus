#pragma once

#include "tensor_useful_concepts.hpp"
#include "tensor_useful_specializations.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <functional>

namespace tensor_lib
{
	namespace tensor_lib_internal
	{
		template<class ForwardIt> requires std::forward_iterator<ForwardIt>
		inline constexpr ForwardIt _constexpr_uninitialized_value_construct_n(ForwardIt first, std::size_t n)
		{
			ForwardIt current = first;

			try 
			{
				for (; n > 0; (void) ++current, --n) 
				{
					std::construct_at(std::addressof(*current));
				}
				return current;
			}
			catch (...) 
			{
				std::destroy(first, current);
				throw;
			}
		}

	}

	template <typename T>
	class _tensor_common
	{
	public:
		struct iterator;
		struct const_iterator;
	};
	

	template <typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>> requires (Rank != 0u)
	class tensor;

	template <typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>> requires (Rank != 0u)
	class subdimension;

	template <typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>> requires (Rank != 0u)
	class const_subdimension;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	inline constexpr void swap(tensor<T, Rank, allocator_type>& left, tensor<T, Rank, allocator_type>& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	inline constexpr void swap(subdimension<T, Rank, allocator_type>& left, subdimension<T, Rank, allocator_type>& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	inline constexpr void swap(tensor<T, Rank, allocator_type>&& left, tensor<T, Rank, allocator_type>&& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	inline constexpr void swap(subdimension<T, Rank, allocator_type>&& left, subdimension<T, Rank, allocator_type>&& right) noexcept;

	template <typename U, typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	concept is_tensor = std::same_as <U, tensor<T, Rank, allocator_type>>
		|| std::same_as<U, subdimension<T, Rank, allocator_type>>
		|| std::same_as<U, const_subdimension<T, Rank, allocator_type>>;

	template <typename T, size_t Rank, typename allocator_type> requires (Rank != 0u)
	class tensor : public _tensor_common<T>, private allocator_type
	{
	private:
		// Stores the size of each individual dimension of the tensor.
		// Ex: Consider tensor_3d<T>. If _order_of_dimension contains { 3u, 4u, 5u }, it means ours is a tensor of 3x4x5 with a total of 120 elements.
		//
		std::array<size_t, Rank> _order_of_dimension{};

		// This is an optimization. Computed when the object is initialized, it contains the equivalent size for each subdimension.
		// Ex: Consider tensor_3d<T>. If _order_of_dimension contains { 3u, 4u, 5u }, 
		// _size_of_subdimension should contain { 120u, 20u, 5u } or { 3u*4u*5u, 4u*5u, 5u }.
		// 
		// This allows methods that rely on the size of our tensor (like "size()") to be O(1) and not have to call std::accumulate() on _order_of_dimension,
		// each time we need the size of a certain dimension.
		//
		std::array<size_t, Rank> _size_of_subdimension{};

		// Dynamically allocated data buffer.
		//
		T* _data = nullptr;

		constexpr allocator_type& get_allocator() noexcept
		{
			return *static_cast<allocator_type*>(this);
		}

		using allocator_type_traits = std::allocator_traits<allocator_type>;

		static constexpr bool no_throw_default_construction = std::is_nothrow_default_constructible_v<T>;
		static constexpr bool no_throw_destructible = std::is_nothrow_destructible_v<T>;
		static constexpr bool no_throw_copyable = std::is_nothrow_copy_assignable_v<T>;

	private:

		template<typename ForwardIt, typename ... Args> 
			requires std::forward_iterator<ForwardIt> 
			&& std::constructible_from<std::iter_value_t<ForwardIt>, Args...>
		inline constexpr void uninitialized_fill(ForwardIt first, ForwardIt last, const Args& ... args)
		{
			using V = std::iter_value_t<ForwardIt>;
			ForwardIt current = first;

			try 
			{
				for (; current != last; ++current) 
				{
					::new (const_cast<void*>(static_cast<const volatile void*>(std::addressof(*current)))) V(args...);
				}
			}
			catch (...) 
			{
				for (; first != current; ++first) 
				{
					first->~V();
				}
				throw;
			}
		}

		template <size_t Rank_index> requires (Rank_index > 2u)
		inline constexpr void _construct_order_array(const useful_specializations::nested_initializer_list_t<T, Rank_index>& data) 
		{
			const auto data_size = data.size();

			if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
			{
				throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");
			}

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				_construct_order_array<Rank_index - 1>(init_list);
			}
		}

		template <size_t Rank_index> requires (Rank_index == 1u)
		inline constexpr void _construct_order_array(const std::initializer_list<T>& data) 
		{
			const auto data_size = data.size();

			if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
			{
				throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");
			}

			_order_of_dimension[Rank - Rank_index] = data_size;
		}

		template <size_t Rank_index> requires (Rank_index == 2u)
		inline constexpr void _construct_order_array(const std::initializer_list<std::initializer_list<T>>& data) 
		{
			const auto data_size = data.size();

			if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
			{
				throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");
			}

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				_construct_order_array<Rank_index - 1>(init_list);
			}
		}

		template<size_t Rank_index, typename Last, typename ... Args> requires (std::integral<Last> && (Rank_index == 1u))
		inline constexpr void _construct_order_array_and_forward_rest(const Last last, const Args& ... data)
		{
			if (last == 0)
			{
				std::fill_n(_order_of_dimension.begin(), Rank, 0u);
				std::fill_n(_size_of_subdimension.begin(), Rank, 0u);
				return;
			}

			_order_of_dimension[Rank - Rank_index] = static_cast<size_t>(last);

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());

			try 
			{
				uninitialized_fill(_data, _data + size_of_current_tensor(), data...);
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				throw;
			}
			
		}

		template<size_t Rank_index, typename First, typename ... Args> requires (std::integral<First> && (Rank_index != 1u))
		inline constexpr void _construct_order_array_and_forward_rest(const First first, const Args& ... args)
		{
			if (first == 0)
			{
				std::fill_n(_order_of_dimension.begin(), Rank, 0u);
				std::fill_n(_size_of_subdimension.begin(), Rank, 0u);
				return;
			}

			_order_of_dimension[Rank - Rank_index] = static_cast<size_t>(first);

			_construct_order_array_and_forward_rest<Rank_index - 1, Args...>(args...);
		}

		template <typename First, typename ... Args>
		inline constexpr bool _are_same_size(const First& first, const Args& ... tensors) noexcept
		{
			return (std::equal(first.get_ranks().begin(), first.get_ranks().end(), tensors.get_ranks().begin(), tensors.get_ranks().end()) && ...);
		}

		template <size_t Index, typename Last>
		inline constexpr void _assign_subdimensions(const Last& last)
		{
			(*this)[Index].replace(last);
		}

		template <size_t Index, typename First, typename ... Args>
		inline constexpr void _assign_subdimensions(const First& first, const Args& ... tensors) 
		{
			(*this)[Index].replace(first);
			_assign_subdimensions<Index + 1, Args...>(tensors...);
		}

	public:

		friend class subdimension<T, Rank, allocator_type>;
		friend class const_subdimension<T, Rank, allocator_type>;

		friend constexpr void swap<T, Rank, allocator_type>(tensor& left, tensor& right) noexcept;
		friend constexpr void swap<T, Rank, allocator_type>(tensor&& left, tensor&& right) noexcept;

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		// All tensor() constructors take a series of unsigned, non-zero, integeres that reprezent
		// the sizes of each dimension, be it as an array or initializer_list.
		//

		inline constexpr tensor(const allocator_type& allocator = allocator_type{})
			: allocator_type { allocator }
			, _order_of_dimension{ {} }
			, _size_of_subdimension{ {} }
			, _data { nullptr }
		{

		}

		template<typename... Sizes> requires (sizeof...(Sizes) == Rank) && useful_concepts::integrals<Sizes...>
		inline constexpr tensor(const Sizes ... sizes)
			: _order_of_dimension{ {} }
			, _size_of_subdimension{ {} }
			, _data { nullptr }
		{
			if (!(sizes && ...))
			{
				return;
			}
			else
			{
				std::array<std::size_t, Rank> temp_order_of_dimension { static_cast<std::size_t>(sizes)... };
				std::array<std::size_t, Rank> temp_size_of_subdimension;
				std::partial_sum(temp_order_of_dimension.crbegin(), temp_order_of_dimension.crend(), temp_size_of_subdimension.rbegin(), std::multiplies<size_t>());
				_data = allocator_type_traits::allocate(get_allocator(), temp_size_of_subdimension[0]);
				try
				{
					if (std::is_constant_evaluated())
					{
						tensor_lib_internal::_constexpr_uninitialized_value_construct_n(&_data[0], temp_size_of_subdimension.front());
					}
					else
					{
						std::uninitialized_default_construct_n(&_data[0], temp_size_of_subdimension.front());
					}
				}
				catch (...)
				{
					allocator_type_traits::deallocate(get_allocator(), _data, temp_size_of_subdimension.front());
				}
				std::copy_n(temp_order_of_dimension.cbegin(), Rank, _order_of_dimension.begin());
				std::copy_n(temp_size_of_subdimension.cbegin(), Rank, _size_of_subdimension.begin());
			}
		}

		inline constexpr tensor(tensor&& other) noexcept
			: allocator_type { other.get_allocator() }
			, _order_of_dimension(std::exchange(other._order_of_dimension, {}))
			, _size_of_subdimension(std::exchange(other._size_of_subdimension, {}))
			, _data(std::exchange(other._data, nullptr))
		{

		}

		inline constexpr tensor(tensor&& other, const allocator_type& allocator)
			: allocator_type { allocator }
		{
			if (get_allocator() == other.get_allocator())
			{
				_order_of_dimension = std::exchange(other._order_of_dimension, {});
				_size_of_subdimension = std::exchange(other._size_of_subdimension, {});
				_data = std::exchange(other._data, nullptr);
			}
			else
			{
				_order_of_dimension = other._order_of_dimension;
				_size_of_subdimension = other._size_of_subdimension;
				_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());

				try
				{
					std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), _data);
				}
				catch (...)
				{
					allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
					throw;
				}
			}
		}

		template<typename U> requires ((Rank == 1u) && std::is_constructible_v<T, U>)
		inline constexpr tensor(const std::initializer_list<U>& data, const allocator_type& allocator = allocator_type{})
			: allocator_type { allocator }
			, _order_of_dimension { data.size() }
			, _size_of_subdimension { data.size() }
		{
			_data = allocator_type_traits::allocate(get_allocator(), data.size());

			try
			{
				std::uninitialized_copy_n(data.begin(), size_of_current_tensor(), _data);
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				throw;
			}
		}

		inline constexpr tensor(const useful_specializations::nested_initializer_list_t<T, Rank>& data, const allocator_type& allocator = allocator_type{} ) requires (Rank > 1u)
			: allocator_type { allocator }
		{
			_construct_order_array<Rank>(data);
			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());

			try
			{
				if (std::is_constant_evaluated())
				{
					tensor_lib_internal::_constexpr_uninitialized_value_construct_n(_data, size_of_current_tensor());
				}
				else
				{
					std::uninitialized_default_construct_n(_data, size_of_current_tensor());
				}
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				throw;
			}
			*this = data; 
		}

		inline constexpr tensor(const tensor& other)
			: allocator_type { other.get_allocator() }
			, _order_of_dimension(other._order_of_dimension)
			, _size_of_subdimension(other._size_of_subdimension)
			, _data(allocator_type_traits::allocate(get_allocator(), other.size_of_current_tensor()))
		{
			try
			{
				std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), _data);
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				throw;
			}
		}

		inline constexpr tensor(const subdimension<T, Rank>& subdimension, const allocator_type& allocator = allocator_type{})
			: allocator_type { allocator }
			, _data(allocator_type_traits::allocate(get_allocator(), subdimension.size_of_current_tensor()))
		{
			try
			{
				std::uninitialized_copy_n(subdimension.cbegin(), subdimension.size_of_current_tensor(), _data);
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, subdimension.size_of_current_tensor());
				throw;
			}
			std::copy_n(subdimension._order_of_dimension.begin(), Rank, _order_of_dimension.begin());
			std::copy_n(subdimension._size_of_subdimension.begin(), Rank, _size_of_subdimension.begin());
			
		}

		template<typename ... Args> requires (sizeof...(Args) > Rank && !(is_tensor<Args, T, Rank - 1, allocator_type> && ...))
			inline constexpr tensor(const Args& ... args)
		{
			_construct_order_array_and_forward_rest<Rank, Args...>(args...);
		}

		template<typename First, typename ... Args> requires (is_tensor<First, T, Rank - 1, allocator_type>) && (is_tensor<Args, T, Rank - 1, allocator_type> && ...) && (sizeof...(Args) > 0)
		inline constexpr tensor(const First& first, const Args& ... tensors)
		{
			if (!_are_same_size(first, tensors...))
			{
				throw std::runtime_error("Can't constructor tensor from tensors of different sizes!");
			}

			_order_of_dimension[0] = sizeof...(Args) + 1;
			std::copy_n(first.get_ranks().begin(), Rank - 1, &_order_of_dimension[1]);

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			
			try
			{
				_assign_subdimensions<0, First, Args...>(first, tensors...);
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				throw;
			}
		}

		inline constexpr auto& operator= (const tensor& other)
		{
			if (!std::is_fundamental_v<T>)
			{
				std::destroy_n(_data, size_of_current_tensor());
			}

			allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());

			_order_of_dimension = other.get_ranks();
			_size_of_subdimension = other.get_sizes();

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), &_data[0]);

			return *this;
		}

		template <typename Tensor_Type>
		inline constexpr auto& operator= (const Tensor_Type& other) requires (is_tensor<Tensor_Type, T, Rank, allocator_type> && !std::is_same_v<Tensor_Type, tensor>)
		{
			if (size_of_current_tensor() != other.size_of_current_tensor())
			{
				if (size_of_current_tensor())
				{
					if (!std::is_fundamental_v<T>)
					{
						std::destroy_n(_data, size_of_current_tensor());
					}

					allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				}

				T* temp_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());

				try 
				{
					std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), temp_data);
				}
				catch (...)
				{
					allocator_type_traits::deallocate(get_allocator(), temp_data, size_of_current_tensor());
				}

				_data = temp_data;
				
			}
			else
			{
				std::copy_n(other.cbegin(), size_of_current_tensor(), _data);
			}

			_order_of_dimension = other.get_ranks();
			_size_of_subdimension = other.get_sizes();

			return *this;
		}

		inline constexpr auto& operator= (tensor&& other) noexcept
		{
			if (this != std::addressof(other))
			{
				if (size_of_current_tensor())
				{
					if constexpr (!std::is_fundamental_v<T>)
					{
						std::destroy_n(_data, size_of_current_tensor());
					}

					allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				}

				_data = std::exchange(other._data, nullptr);
				_order_of_dimension = std::exchange(other._order_of_dimension, {});
				_size_of_subdimension = std::exchange(other._size_of_subdimension, {});
			}
			return *this;
		}

		template<typename Tensor_Type> requires (is_tensor<Tensor_Type, T, Rank, allocator_type>)
		inline constexpr auto& replace(const Tensor_Type& other)
		{
			if (!std::equal(_order_of_dimension.begin(), _order_of_dimension.end(), other._order_of_dimension.begin(), other._order_of_dimension.end()))
			{
				throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");
			}

			std::copy_n(&other._data[0], size_of_current_tensor(), &_data[0]);

			return *this;
		}

		template <typename First, typename ... Args> requires (is_tensor<First, T, Rank - 1, allocator_type>) && (is_tensor<Args, T, Rank - 1, allocator_type> && ...) && (sizeof...(Args) > 0)
		inline constexpr auto& replace(const First& first, const Args& ... tensors)
		{
			if (!_are_same_size(first, tensors...))
			{
				throw std::runtime_error("Can't constructor tensor from tensors of different sizes!");
			}

			std::array<size_t, Rank> temp_order_of_dimension;
			std::array<size_t, Rank> temp_size_of_subdimension;
			T* temp_data;

			temp_order_of_dimension[0] = sizeof...(Args) + 1;
			std::copy_n(first.get_ranks().begin(), Rank - 1, temp_order_of_dimension.begin() + 1);

			std::partial_sum(temp_order_of_dimension.crbegin(), temp_order_of_dimension.crend(), temp_size_of_subdimension.rbegin(), std::multiplies<size_t>());

			temp_data = allocator_type_traits::allocate(get_allocator(), temp_size_of_subdimension[0]);

			ptrdiff_t index = 0;

			auto assign_from_tensor = [temp_data, &index](const auto& tensor)
			{
				for (const auto& val : tensor)
				{
					std::construct_at(&temp_data[index], val);
					++index;
				}
			};

			try 
			{
				assign_from_tensor(first);
				((assign_from_tensor(tensors)), ...);
			}
			catch (...)
			{
				--index;
				while (index >= 0)
				{
					std::destroy_at(&temp_data[index]);
					--index;
				}
				allocator_type_traits::deallocate(get_allocator(), temp_data, size_of_current_tensor());
				throw;
			}

			if (size_of_current_tensor())
			{
				if (!std::is_fundamental_v<T>)
				{
					std::destroy_n(_data, size_of_current_tensor());
				}

				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
			}

			_data = temp_data;
			std::copy_n(temp_order_of_dimension.cbegin(), Rank, _order_of_dimension.begin());
			std::copy_n(temp_size_of_subdimension.cbegin(), Rank, _size_of_subdimension.begin());

			return *this;
		}

		template<typename Iterator> requires(std::forward_iterator<Iterator> && std::is_convertible_v<std::iter_value_t<Iterator>, T>)
		inline constexpr auto& replace(Iterator first, Iterator last)
		{
			const auto size = static_cast<size_t>(std::distance(first, last));

			if (size != size_of_current_tensor())
				throw std::runtime_error("Range different in size than tensor!");

			std::copy_n(first, size, begin());

			return (*this);
		}

		inline constexpr auto& operator=(const useful_specializations::nested_initializer_list_t<T, Rank>& data) requires (Rank > 2u)
		{
			if (order_of_current_dimension() != data.size())
			{
				throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		inline constexpr auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) requires (Rank == 2)
		{
			if (order_of_current_dimension() != data.size())
			{
				throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		inline constexpr auto& operator=(const std::initializer_list<T>& data)
		{
			if (size_of_current_tensor() != data.size())
			{
				throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");
			}

			std::copy_n(data.begin(), data.size(), begin());

			return (*this);
		}

		template<typename... Sizes> requires (sizeof...(Sizes) == Rank) && useful_concepts::integrals<Sizes...>
		inline constexpr void resize(const Sizes ... new_sizes)
		{
			const auto old_size = size_of_current_tensor();

			if (old_size)
			{
				std::destroy_n(_data, size_of_current_tensor());
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				_data = nullptr;
			}

			if (!(new_sizes && ...))
			{
				std::fill_n(_order_of_dimension.begin(), Rank, 0u);
				std::fill_n(_size_of_subdimension.begin(), Rank, 0u);
				return;
			}

			_order_of_dimension = { static_cast<size_t>(new_sizes)... };

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			try
			{
				if (std::is_constant_evaluated())
				{
					tensor_lib_internal::_constexpr_uninitialized_value_construct_n(&_data[0], size_of_current_tensor());
				}
				else
				{
					std::uninitialized_default_construct_n(&_data[0], size_of_current_tensor());
				}
			}
			catch (...)
			{
				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
				_data = nullptr;
				std::fill_n(_order_of_dimension.begin(), Rank, 0u);
				std::fill_n(_size_of_subdimension.begin(), Rank, 0u);
				return;
			}
		}

		inline constexpr auto operator[] (const size_t index) noexcept requires (Rank > 1u)
		{
			return subdimension<T, Rank - 1>
				(
					_order_of_dimension.begin() + 1,
					_order_of_dimension.end(),
					_size_of_subdimension.begin() + 1,
					_size_of_subdimension.end(),
					begin() + index * _size_of_subdimension[1],
					begin() + index * _size_of_subdimension[1] + _size_of_subdimension[1]
					);
		}

		inline constexpr auto operator[] (const size_t index) const noexcept requires (Rank > 1u)
		{
			return const_subdimension<T, Rank - 1>
				(
					_order_of_dimension.begin() + 1,
					_order_of_dimension.end(),
					_size_of_subdimension.begin() + 1,
					_size_of_subdimension.end(),
					cbegin() + index * _size_of_subdimension[1],
					cbegin() + index * _size_of_subdimension[1] + _size_of_subdimension[1]
					);
		}

		inline constexpr auto& operator[] (const size_t index) noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		inline constexpr auto& operator[] (const size_t index) const noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		inline constexpr iterator begin() noexcept
		{
			return iterator(&_data[0]);
		}

		inline constexpr const_iterator begin() const noexcept
		{
			return const_iterator(&_data[0]);
		}

		inline constexpr const_iterator cbegin() const noexcept
		{
			return const_iterator(&_data[0]);
		}

		inline constexpr iterator end() noexcept
		{
			return iterator(&_data[size_of_current_tensor()]);
		}

		inline constexpr const_iterator end() const noexcept
		{
			return const_iterator(&_data[size_of_current_tensor()]);
		}

		inline constexpr const_iterator cend() const noexcept
		{
			return const_iterator(&_data[size_of_current_tensor()]);
		}

		inline constexpr auto get_sizes() const noexcept
		{
			return std::span<const size_t, Rank>(_size_of_subdimension);
		}

		inline constexpr auto get_ranks() const noexcept
		{
			return std::span<const size_t, Rank>(_order_of_dimension);
		}

		inline constexpr size_t order_of_dimension(const size_t& index) const noexcept
		{
			return _order_of_dimension[index];
		}

		inline constexpr size_t size_of_subdimension(const size_t& index) const noexcept
		{
			return _size_of_subdimension[index];
		}

		inline constexpr size_t order_of_current_dimension() const noexcept
		{
			return _order_of_dimension[0];
		}

		inline constexpr size_t size_of_current_tensor() const noexcept
		{
			return _size_of_subdimension[0];
		}

		inline constexpr bool empty() const noexcept
		{
			return _size_of_subdimension[0] == 0;
		}

		inline consteval bool is_matrix() const noexcept
		{
			return (Rank == 2);
		}

		inline constexpr T* data() noexcept
		{
			return _data;
		}

		inline constexpr const T* data() const noexcept
		{
			return _data;
		}

		inline constexpr ~tensor()
		{
			std::destroy_n(_data, size_of_current_tensor());
			allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());
		}
	};

	template <typename T, size_t Rank, typename allocator_type> requires (Rank != 0u)
	class const_subdimension : public _tensor_common<T>
	{
		using ConstSourceSizeOfDimensionArraySpan = std::span<const size_t, Rank>;
		using ConstSourceSizeOfSubdimensionArraySpan = std::span<const size_t, Rank>;
		using ConstSourceDataSpan = std::span<const T>;

	private:
		ConstSourceSizeOfDimensionArraySpan	_order_of_dimension;
		ConstSourceSizeOfSubdimensionArraySpan _size_of_subdimension;
		ConstSourceDataSpan	_data;

		static constexpr bool no_throw_default_construction = std::is_nothrow_default_constructible_v<T>;
		static constexpr bool no_throw_destructible = std::is_nothrow_destructible_v<T>;
		static constexpr bool no_throw_copyable = std::is_nothrow_copy_assignable_v<T>;

	public:

		friend class subdimension<T, Rank, allocator_type>;
		friend class tensor<T, Rank, allocator_type>;

		using iterator = typename _tensor_common<T>::const_iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		inline constexpr const_subdimension() = delete;
		inline constexpr const_subdimension(const_subdimension&&) noexcept = default;
		inline constexpr const_subdimension(const const_subdimension&) noexcept = default;

		inline constexpr const_subdimension(const subdimension<T, Rank, allocator_type>& other) noexcept :
			_order_of_dimension{ other._order_of_dimension },
			_size_of_subdimension{ other._size_of_subdimension },
			_data{ other._data }
		{

		}

		template<typename T1, typename T2, typename T3>
			requires std::constructible_from<decltype(_order_of_dimension), T1>
			&& std::constructible_from<decltype(_size_of_subdimension), T2>
			&& std::constructible_from<decltype(_data), T3>
		inline constexpr const_subdimension(T1&& param_1, T2&& param_2, T3&& param_3) noexcept
			: _order_of_dimension(std::forward<T1>(param_1))
			, _size_of_subdimension(std::forward<T2>(param_2))
			, _data(std::forward<T3>(param_3))
		{

		}

		template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
			requires std::constructible_from<decltype(_order_of_dimension), T1, T2>
			&& std::constructible_from<decltype(_size_of_subdimension), T3, T4>
			&& std::constructible_from<decltype(_data), T5, T6>
		inline constexpr const_subdimension(T1&& param_1, T2&& param_2, T3&& param_3, T4&& param_4, T5&& param_5, T6&& param_6) noexcept
			: _order_of_dimension(std::forward<T1>(param_1), std::forward<T2>(param_2))
			, _size_of_subdimension(std::forward<T3>(param_3), std::forward<T4>(param_4))
			, _data(std::forward<T5>(param_5), std::forward<T6>(param_6))
		{

		}

		template<typename Tensor>
			requires std::is_same_v<Tensor, tensor<const T, Rank, allocator_type>>
		inline constexpr const_subdimension(const Tensor& tsor) noexcept :
			_order_of_dimension{ tsor._order_of_dimension.begin(), tsor._order_of_dimension.end() },
			_size_of_subdimension{ tsor._size_of_subdimension.begin(), tsor._size_of_subdimension.end() },
			_data{ tsor.begin(), tsor.end() }
		{

		}

		inline constexpr auto& operator=(const const_subdimension& other) noexcept
		{
			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;
			_data = other._data;

			return *this;
		}

		inline constexpr auto operator[] (const size_t index) const noexcept requires (Rank > 1u)
		{
			return const_subdimension<T, Rank - 1>
				(
				_order_of_dimension.begin() + 1,
				_order_of_dimension.end(),
				_size_of_subdimension.begin() + 1,
				_size_of_subdimension.end(),
				cbegin() + index * _size_of_subdimension[1],
				cbegin() + index * _size_of_subdimension[1] + _size_of_subdimension[1]
				);
		}

		inline constexpr const T& operator[] (const size_t index) const noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		inline constexpr auto begin() const noexcept
		{
			return const_iterator(_data.data());
		}

		inline constexpr auto end() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		inline constexpr auto cbegin() const noexcept
		{
			return const_iterator(_data.data());
		}

		inline constexpr auto cend() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		inline constexpr auto rank() const noexcept
		{
			return _order_of_dimension;
		}

		inline constexpr size_t order_of_dimension(const size_t& index) const noexcept
		{
			return _order_of_dimension[index];
		}

		inline constexpr size_t size_of_subdimension(const size_t& index) const noexcept
		{
			return _size_of_subdimension[index];
		}

		inline constexpr size_t order_of_current_dimension() const noexcept
		{
			return _order_of_dimension[0];
		}

		inline constexpr size_t size_of_current_tensor() const noexcept
		{
			return _size_of_subdimension[0];
		}

		inline constexpr auto get_sizes() const noexcept
		{
			return std::span<const size_t, Rank>(_size_of_subdimension);
		}

		inline constexpr auto get_ranks() const noexcept
		{
			return std::span<const size_t, Rank>(_order_of_dimension);
		}
		
		inline constexpr bool empty() const noexcept
		{
			return _size_of_subdimension[0] == 0;
		}

		inline constexpr const T* data() const noexcept
		{
			return std::to_address(begin());
		}

		inline consteval bool is_matrix() const noexcept
		{
			return (Rank == 2);
		}

		inline constexpr bool is_square_matrix() const noexcept requires (Rank == 2u)
		{
			return (_size_of_subdimension[0] == _size_of_subdimension[1]);
		}
	};

	template <typename T, size_t Rank, typename allocator_type> requires (Rank != 0u)
	class subdimension : public _tensor_common<T>
	{
		using SourceSizeOfDimensionArraySpan = std::span<size_t, Rank>;
		using SourceSizeOfSubdimensionArraySpan = std::span<size_t, Rank>;
		using SourceDataSpan = std::span<T>;

	private:
		SourceSizeOfDimensionArraySpan              _order_of_dimension;
		SourceSizeOfSubdimensionArraySpan           _size_of_subdimension;
		SourceDataSpan                              _data;

		static constexpr bool no_throw_default_construction = std::is_nothrow_default_constructible_v<T>;
		static constexpr bool no_throw_destructible = std::is_nothrow_destructible_v<T>;
		static constexpr bool no_throw_copyable = std::is_nothrow_copy_assignable_v<T>;

		template <typename First, typename ... Args>
		inline constexpr bool _are_same_size(const First& first, const Args& ... tensors) noexcept
		{
			return (std::equal(first.get_ranks().begin(), first.get_ranks().end(), tensors.get_ranks().begin(), tensors.get_ranks().end()) && ...);
		}

		template <size_t Index, typename Last>
		inline constexpr void _assign_subdimensions(const Last& last)
		{
			(*this)[Index].replace(last);
		}

		template <size_t Index, typename First, typename ... Args>
		inline constexpr void _assign_subdimensions(const First& first, const Args& ... tensors) noexcept
		{
			(*this)[Index].replace(first);
			_assign_subdimensions<Index + 1, Args...>(tensors...);
		}

	public:
		friend class const_subdimension<T, Rank, allocator_type>;
		friend class tensor<T, Rank, allocator_type>;
		friend class tensor<T, Rank + 1u, allocator_type>;
		friend class subdimension<T, Rank + 1, allocator_type>;
		friend class tensor<T, (Rank > 1u) ? (Rank - 1u) : 1u, allocator_type>;

		friend constexpr void swap<T, Rank, allocator_type>(subdimension&, subdimension&) noexcept;
		friend constexpr void swap<T, Rank, allocator_type>(subdimension&&, subdimension&&) noexcept;

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		inline constexpr subdimension() = delete;
		inline constexpr subdimension(subdimension&&) noexcept = delete;
		inline constexpr subdimension(const subdimension&) noexcept = default;
		inline constexpr subdimension(const const_subdimension<T, Rank>&) noexcept = delete;

		template<typename T1, typename T2, typename T3>
		requires std::constructible_from<decltype(_order_of_dimension), T1>
			&& std::constructible_from<decltype(_size_of_subdimension), T2>
			&& std::constructible_from<decltype(_data), T3>
		inline constexpr subdimension(T1&& param_1, T2&& param_2, T3&& param_3) noexcept
			: _order_of_dimension{ std::forward<T1>(param_1) }
			, _size_of_subdimension{ std::forward<T2>(param_2) }
			, _data{ std::forward<T3>(param_3) }
		{

		}

		template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
		requires std::constructible_from<decltype(_order_of_dimension), T1, T2>
			&& std::constructible_from<decltype(_size_of_subdimension), T3, T4>
			&& std::constructible_from<decltype(_data), T5, T6>
		inline constexpr subdimension(T1&& param_1, T2&& param_2, T3&& param_3, T4&& param_4, T5&& param_5, T6&& param_6) noexcept
			: _order_of_dimension(std::forward<T1>(param_1), std::forward<T2>(param_2))
			, _size_of_subdimension(std::forward<T3>(param_3), std::forward<T4>(param_4))
			, _data(std::forward<T5>(param_5), std::forward<T6>(param_6))
		{

		}

		inline constexpr subdimension(tensor<T, Rank>& mat) noexcept
			: _order_of_dimension{ mat._order_of_dimension.begin(), mat._order_of_dimension.end() }
			, _size_of_subdimension{ mat._size_of_subdimension.begin(), mat._size_of_subdimension.end() }
			, _data{ mat.begin(), mat.end() }
		{

		}

		inline constexpr auto& operator=(const subdimension& other) noexcept
		{
			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;
			_data = other._data;

			return *this;
		}

		template<typename Iterator> requires (std::forward_iterator<Iterator> && std::is_constructible_v<typename std::iterator_traits<Iterator>::value_type, T>)
		inline constexpr auto& replace(Iterator first, Iterator last)
		{
			const auto size = static_cast<size_t>(std::distance(first, last));

			if (size != size_of_current_tensor())
			{
				throw std::runtime_error("Range differs in size with tensor!");
			}

			std::copy_n(first, size, begin());

			return *this;
		}

		template<typename Tensor_Type> requires (is_tensor<Tensor_Type, T, Rank, allocator_type>)
		inline constexpr auto& replace(const Tensor_Type& other)
		{
			if (!std::equal(_order_of_dimension.begin(), _order_of_dimension.end(), other._order_of_dimension.begin(), other._order_of_dimension.end()))
			{
				throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");
			}

			std::copy_n(other.cbegin(), size_of_current_tensor(), begin());

			return *this;
		}

		template<typename ... Tensors> requires ((sizeof...(Tensors) > 1) && (is_tensor<Tensors, T, Rank - 1, allocator_type> && ...))
		inline constexpr auto& replace(const Tensors& ... tensors)
		{
			if (!_are_same_size((*this)[0], tensors...))
			{
				throw std::runtime_error("Size of tensor we take values from must match the size of current subdimension");
			}

			_assign_subdimensions<0, Tensors...>(tensors...);

			return (*this);
		}

		inline constexpr auto& operator=(const useful_specializations::nested_initializer_list_t<T, Rank>& data) requires (Rank > 2u)
		{
			if (order_of_current_dimension() != data.size())
			{
				throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		inline constexpr auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) requires (Rank == 2u)
		{
			if (order_of_current_dimension() != data.size())
			{
				throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		inline constexpr auto& operator=(const std::initializer_list<T>& data)
		{
			if (size_of_current_tensor() != data.size())
			{
				throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");
			}

			std::copy_n(data.begin(), data.size(), begin());

			return (*this);
		}

		inline constexpr auto& operator=(const tensor<T, Rank, allocator_type>& tensor)
		{
			if (!std::equal(_order_of_dimension.begin(), _order_of_dimension.end(), tensor.get_ranks().begin(), tensor.get_ranks().end()))
			{
				throw std::runtime_error("Size of tensor doesn't match size of subdimension!");
			}

			std::copy_n(tensor.cbegin(), size_of_current_tensor(), begin());

			return (*this);
		}

		inline constexpr auto operator[] (const size_t index) noexcept requires (Rank > 1u)
		{
			return subdimension<T, Rank - 1>
				(
					_order_of_dimension.begin() + 1,
					_order_of_dimension.end(),
					_size_of_subdimension.begin() + 1,
					_size_of_subdimension.end(),
					begin() + index * _size_of_subdimension[1],
					begin() + index * _size_of_subdimension[1] + _size_of_subdimension[1]
					);
		}

		inline constexpr auto operator[] (const size_t index) const noexcept requires (Rank > 1u)
		{
			return const_subdimension<T, Rank - 1>
				(
					_order_of_dimension.begin() + 1,
					_order_of_dimension.end(),
					_size_of_subdimension.begin() + 1,
					_size_of_subdimension.end(),
					cbegin() + index * _size_of_subdimension[1],
					cbegin() + index * _size_of_subdimension[1] + _size_of_subdimension[1]
					);
		}

		inline constexpr const T& operator[] (const size_t index) const noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		inline constexpr T& operator[] (const size_t index) noexcept requires(Rank == 1u)
		{
			return _data[index];
		}

		inline constexpr auto begin() noexcept
		{
			return iterator(_data.data());
		}

		inline constexpr auto begin() const noexcept
		{
			return const_iterator(_data.data());
		}

		inline constexpr auto cbegin() const noexcept
		{
			return const_iterator(_data.data());
		}

		inline constexpr auto end() noexcept
		{
			return iterator(std::to_address(_data.end()));
		}

		inline constexpr auto end() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		inline constexpr auto cend() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		inline constexpr size_t& order_of_dimension(const size_t& index) const noexcept
		{
			return _order_of_dimension[index];
		}

		inline constexpr size_t& size_of_subdimension(const size_t& index) const noexcept
		{
			return _size_of_subdimension[index];
		}

		inline constexpr size_t& order_of_current_dimension() const noexcept
		{
			return _order_of_dimension[0];
		}

		inline constexpr size_t& size_of_current_tensor() const noexcept
		{
			return _size_of_subdimension[0];
		}

		inline constexpr bool empty() const noexcept
		{
			return _size_of_subdimension[0] == 0;
		}

		inline constexpr T* data() noexcept
		{
			return _data.data();
		}

		inline constexpr const T* data() const noexcept
		{
			return _data.data();
		}

		inline consteval bool is_matrix() const noexcept
		{
			return (Rank == 2);
		}

		inline constexpr auto get_sizes() const noexcept
		{
			return std::span<const size_t, Rank>(_size_of_subdimension);
		}

		inline constexpr auto get_ranks() const noexcept
		{
			return std::span<const size_t, Rank>(_order_of_dimension);
		}
	};

	template <typename T>
	struct _tensor_common<T>::iterator
	{
		using difference_type = ptrdiff_t;
		using value_type = T;
		using element_type = T;
		using pointer = T*;
		using reference = T&;
		using iterator_category = std::contiguous_iterator_tag;

		inline constexpr iterator() noexcept : ptr{} {};
		inline constexpr iterator(const iterator&) noexcept = default;
		inline constexpr iterator(iterator&&) noexcept = default;
		inline constexpr iterator(pointer other) noexcept : ptr{ other } {}
		inline constexpr iterator(bool) = delete;

		inline constexpr explicit operator reference() const noexcept
		{
			return *ptr;
		}

		inline constexpr iterator& operator=(const iterator&) = default;

		inline constexpr iterator& operator=(iterator&&) = default;

		inline constexpr iterator& operator= (const pointer other) noexcept
		{
			ptr = other;
			return (*this);
		}

		inline constexpr reference operator* () const noexcept
		{
			return *ptr;
		}

		inline constexpr pointer operator-> () const noexcept
		{
			return ptr;
		}

		inline constexpr iterator& operator++ () noexcept
		{
			++ptr;
			return *this;
		}

		inline constexpr iterator operator++(int) noexcept
		{
			return iterator(ptr++);
		}

		inline constexpr iterator& operator-- () noexcept
		{
			--ptr;
			return *this;
		}

		inline constexpr iterator operator--(int) noexcept
		{
			return iterator(ptr--);
		}

		inline constexpr iterator& operator+=(const ptrdiff_t offset) noexcept
		{
			ptr += offset;
			return *this;
		}

		inline constexpr iterator& operator-=(const ptrdiff_t offset) noexcept
		{
			ptr -= offset;
			return *this;
		}

		inline constexpr reference operator[](const size_t offset) const noexcept
		{
			return ptr[offset];
		}

		inline constexpr friend bool operator==(const iterator it_a, const iterator it_b) noexcept
		{
			return it_a.ptr == it_b.ptr;
		}

		inline constexpr friend bool operator!=(const iterator it_a, const iterator it_b) noexcept
		{
			return it_a.ptr != it_b.ptr;
		}

		inline constexpr friend iterator operator+(const iterator it, const size_t offset) noexcept
		{
			T* result = it.ptr + offset;
			return iterator(result);
		}

		inline constexpr friend iterator operator+(const size_t offset, const iterator& it) noexcept
		{
			auto aux = offset + it.ptr;
			return iterator(aux);
		}

		inline constexpr friend iterator operator-(const iterator it, const size_t offset) noexcept
		{
			T* aux = it.ptr - offset;
			return iterator(aux);
		}

		inline constexpr friend difference_type operator-(const iterator a, const iterator b) noexcept
		{
			return a.ptr - b.ptr;
		}

		inline constexpr friend auto operator<=>(const iterator a, const iterator b) noexcept
		{
			return a.ptr <=> b.ptr;
		}

	private:

		pointer ptr;
	};

	template <typename T>
	struct _tensor_common<T>::const_iterator
	{
		using difference_type = ptrdiff_t;
		using value_type = const T;
		using element_type = T;
		using pointer = const T*;
		using reference = const T&;
		using iterator_category = std::contiguous_iterator_tag;

		inline constexpr const_iterator() noexcept : ptr{} {};
		inline constexpr const_iterator(const const_iterator&) noexcept = default;
		inline constexpr const_iterator(const_iterator&&) noexcept = default;
		inline constexpr const_iterator(pointer other) noexcept : ptr{ other } {}

		inline constexpr const_iterator(bool) = delete;
		inline
		constexpr const_iterator(const iterator other) noexcept : ptr{ other.ptr } {}

		inline constexpr explicit operator reference() const noexcept
		{
			return *ptr;
		}

		inline constexpr const_iterator& operator=(const const_iterator&) noexcept = default;

		inline constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

		inline constexpr const_iterator& operator= (const pointer other) noexcept
		{
			ptr = other;
			return (*this);
		}

		inline constexpr reference operator*() const noexcept
		{
			return *ptr;
		}

		inline constexpr pointer operator-> () const noexcept
		{
			return ptr;
		}

		inline constexpr const_iterator& operator++ () noexcept
		{
			++ptr;
			return *this;
		}

		inline constexpr const_iterator operator++(int) noexcept
		{
			return const_iterator(ptr++);
		}

		inline constexpr const_iterator& operator-- () noexcept
		{
			--ptr;
			return *this;
		}

		inline constexpr const_iterator operator--(int) noexcept
		{
			return const_iterator(ptr--);
		}

		inline constexpr const_iterator& operator+=(const size_t offset) noexcept
		{
			ptr += offset;
			return *this;
		}

		inline constexpr const_iterator& operator-=(const size_t offset) noexcept
		{
			ptr -= offset;
			return *this;
		}

		inline constexpr reference operator[](const size_t offset) const noexcept
		{
			return ptr[offset];
		}

		inline constexpr friend bool operator==(const const_iterator it_a, const const_iterator it_b) noexcept
		{
			return it_a.ptr == it_b.ptr;
		}

		inline constexpr friend bool operator==(const const_iterator it, const pointer adr) noexcept
		{
			return it.ptr == adr;
		}

		inline constexpr friend bool operator!=(const const_iterator it_a, const const_iterator it_b) noexcept
		{
			return it_a.ptr != it_b.ptr;
		}

		inline constexpr friend bool operator!=(const const_iterator it, const pointer adr) noexcept
		{
			return it.ptr != adr;
		}

		inline constexpr friend const_iterator operator+(const const_iterator a, const const_iterator b) noexcept
		{
			return const_iterator(a.ptr + b.ptr);
		}

		inline constexpr friend const_iterator operator+(const const_iterator& it, const size_t offset) noexcept
		{
			return const_iterator(it.ptr + offset);
		}

		inline constexpr friend const_iterator operator+(const size_t offset, const const_iterator it) noexcept
		{
			return const_iterator(offset + it.ptr);
		}

		inline constexpr friend const_iterator operator-(const const_iterator it, const size_t offset) noexcept
		{
			return const_iterator(it.ptr - offset);
		}

		inline constexpr friend difference_type operator-(const const_iterator a, const const_iterator b) noexcept
		{
			return a.ptr - b.ptr;
		}

		inline constexpr friend auto operator<=>(const const_iterator a, const const_iterator b) noexcept
		{
			return a.ptr <=> b.ptr;
		}

	private:

		pointer ptr;
	};

	template <typename T, size_t Rank, typename allocator_type>
	inline constexpr void swap(tensor<T, Rank, allocator_type>& left, tensor<T, Rank, allocator_type>& right) noexcept
	{
		std::swap(left._order_of_dimension, right._order_of_dimension);
		std::swap(left._size_of_subdimension, right._size_of_subdimension);
		std::swap(left._data, right._data);
	}

	template <typename T, size_t Rank, typename allocator_type>
	inline constexpr void swap(subdimension<T, Rank, allocator_type>& left, subdimension<T, Rank, allocator_type>& right) noexcept
	{
		if (!std::equal(left._order_of_dimension.begin(), left._order_of_dimension.end(), right._order_of_dimension.begin(), right._order_of_dimension.end()))
		{
			std::exit(-1);
		}

		std::unique_ptr<T[]> aux = std::make_unique<T[]>(left.size_of_current_tensor());

		std::copy(left.cbegin(), left.cend(), &aux[0]);
		std::copy(right.cbegin(), right.cend(), left.begin());
		std::copy(&aux[0], &aux[left.size_of_current_tensor()], right.begin());
	}

	template <typename T, size_t Rank, typename allocator_type>
	inline constexpr void swap(tensor<T, Rank, allocator_type>&& left, tensor<T, Rank, allocator_type>&& right) noexcept
	{
		swap(left, right);
	}

	template <typename T, size_t Rank, typename allocator_type>
	inline constexpr void swap(subdimension<T, Rank, allocator_type>&& left, subdimension<T, Rank, allocator_type>&& right) noexcept
	{
		swap(left, right);
	}

	template <typename T>
	using tensor_1d = tensor<T, 1>;

	template <typename T>
	using tensor_line = tensor<T, 1>;

	template <typename T>
	using tensor_2d = tensor<T, 2>;

	template <typename T>
	using matrix = tensor<T, 2>;

	template <typename T>
	using tensor_3d = tensor<T, 3>;

	template <typename T>
	using cube = tensor<T, 3>;

	template <typename T>
	using tensor_4d = tensor<T, 4>;

	template <typename T>
	using tensor_5d = tensor<T, 5>;
}