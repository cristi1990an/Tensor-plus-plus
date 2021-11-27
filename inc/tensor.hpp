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
	// The TENSORLIB_DEBUGGING constant expression indicates wheather the algorithms 
	// will do extra range checks. Set it to 'false' manually for slightly better performance.
	// Keep in mind though that this will disable certain range checks that keep an instance 
	// of the tensor from entering invalid states (like having a subdimension of size zero). 
	// Only disable it if you know what you're doing.

#ifdef _MSC_VER
#ifdef _DEBUG
	constexpr static bool TENSORLIB_DEBUGGING = true;
#else
	constexpr static bool TENSORLIB_DEBUGGING = false;
#endif // _DEBUG
#else
	constexpr static bool TENSORLIB_DEBUGGING = true;
#endif

	constexpr static bool TENSORLIB_RELEASE = !TENSORLIB_DEBUGGING;

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
	void swap(tensor<T, Rank, allocator_type>& left, tensor<T, Rank, allocator_type>& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(subdimension<T, Rank, allocator_type>& left, subdimension<T, Rank, allocator_type>& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(tensor<T, Rank, allocator_type>&& left, tensor<T, Rank, allocator_type>&& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(subdimension<T, Rank, allocator_type>&& left, subdimension<T, Rank, allocator_type>&& right) noexcept;

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

		allocator_type& get_allocator() noexcept
		{
			return *static_cast<allocator_type*>(this);
		}

		using allocator_type_traits = std::allocator_traits<allocator_type>;

		static constexpr bool no_throw_default_construction = std::is_nothrow_default_constructible_v<T>;
		static constexpr bool no_throw_destructible = std::is_nothrow_destructible_v<T>;
		static constexpr bool no_throw_copyable = std::is_nothrow_copy_assignable_v<T>;

	private:
		template <size_t Rank_index> requires (Rank_index > 2u)
		inline constexpr void _construct_order_array(const useful_specializations::nested_initializer_list_t<T, Rank_index>& data) noexcept(TENSORLIB_RELEASE)
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
				{
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");
				}
			}

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				_construct_order_array<Rank_index - 1>(init_list);
			}
		}

		template <size_t Rank_index> requires (Rank_index == 1u)
		inline constexpr void _construct_order_array(const std::initializer_list<T>& data) noexcept(TENSORLIB_RELEASE)
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
				{
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");
				}
			}

			_order_of_dimension[Rank - Rank_index] = data_size;
		}

		template <size_t Rank_index> requires (Rank_index == 2u)
		inline constexpr void _construct_order_array(const std::initializer_list<std::initializer_list<T>>& data) noexcept(TENSORLIB_RELEASE)
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
				{
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");
				}
			}

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				_construct_order_array<Rank_index - 1>(init_list);
			}
		}

		template<size_t Rank_index, typename Last, typename ... Args> requires (std::integral<Last> && (Rank_index == 1u))
		inline constexpr void _construct_order_array_and_forward_rest(const Last last, const Args& ... data) noexcept(!TENSORLIB_DEBUGGING && std::is_nothrow_constructible_v<T, Args...>)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (last == 0)
					throw std::runtime_error("Size of subdimension cannot be zero!");
			}

			_order_of_dimension[Rank - Rank_index] = static_cast<size_t>(last);

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			//std::uninitialized_fill_n(&_data[0], size_of_current_tensor(), T(data...));
			for (size_t index = 0; index != size_of_current_tensor(); ++index) // this is apparently significantly faster in this case...
			{
				std::construct_at(&_data[index], data...);
			}
		}

		template<size_t Rank_index, typename First, typename ... Args> requires (std::integral<First> && (Rank_index != 1u))
		inline constexpr void _construct_order_array_and_forward_rest(const First first, const Args& ... args) noexcept(TENSORLIB_RELEASE&& std::is_nothrow_constructible_v<T, Args...>)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (first == 0)
					throw std::runtime_error("Size of subdimension cannot be zero!");
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
		inline constexpr void _assign_subdimensions(const First& first, const Args& ... tensors) noexcept
		{
			(*this)[Index].replace(first);
			_assign_subdimensions<Index + 1, Args...>(tensors...);
		}

	public:

		friend class subdimension<T, Rank, allocator_type>;
		friend class const_subdimension<T, Rank, allocator_type>;

		friend void swap<T, Rank, allocator_type>(tensor& left, tensor& right) noexcept;
		friend void swap<T, Rank, allocator_type>(tensor&& left, tensor&& right) noexcept; 

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		// All tensor() constructors take a series of unsigned, non-zero, integeres that reprezent
		// the sizes of each dimension, be it as an array or initializer_list.
		//

		constexpr tensor(const allocator_type& allocator = allocator_type{})
			: allocator_type { allocator }
			, _order_of_dimension(useful_specializations::array_filled_with<size_t, Rank>(1u))
			, _size_of_subdimension(useful_specializations::array_filled_with<size_t, Rank>(1u))
			, _data(allocator_type_traits::allocate(get_allocator(), 1u))
		{
			std::uninitialized_default_construct_n(&_data[0], 1u);
		}

		template<typename... Sizes> requires (sizeof...(Sizes) == Rank) && useful_concepts::integrals<Sizes...> && TENSORLIB_RELEASE
		tensor(const Sizes ... sizes) 
		: _order_of_dimension{ static_cast<size_t>(sizes)... }
		{
			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());
			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			std::uninitialized_default_construct_n(&_data[0], size_of_current_tensor());
		}

		template<typename... Sizes> requires ((sizeof...(Sizes) == Rank) && useful_concepts::integrals<Sizes...> && TENSORLIB_DEBUGGING)
		constexpr tensor(const Sizes ... sizes) 
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				const bool contains_zero = ((sizes == 0) || ...);

				if (contains_zero)
				{
					throw std::runtime_error("Size of subdimension cannot be zero!");
				}
			}

			_order_of_dimension = { static_cast<size_t>(sizes)... };
			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			std::uninitialized_default_construct_n(&_data[0], size_of_current_tensor());
		}

		constexpr tensor(tensor&& other)
			: allocator_type { other.get_allocator() }
			, _order_of_dimension(std::move(other._order_of_dimension))
			, _size_of_subdimension(std::move(other._size_of_subdimension))
			, _data(std::move(other._data))
		{
			std::fill_n(other._order_of_dimension.begin(), Rank, 1u);
			std::fill_n(other._size_of_subdimension.begin(), Rank, 1u);

			other._data = allocator_type_traits::allocate(other.get_allocator(), 1u);
			std::uninitialized_default_construct_n(&other._data[0], 1u);
		}

		template<typename U> requires ((Rank == 1u) && std::is_constructible_v<T, U>)
		constexpr tensor(const std::initializer_list<U>& data, const allocator_type& allocator = allocator_type{})
			: allocator_type { allocator }
		{
			_order_of_dimension[0] = data.size();
			_size_of_subdimension[0] = data.size();

			_data = allocator_type_traits::allocate(get_allocator(), data.size());
			std::uninitialized_copy_n(data.begin(), size_of_current_tensor(), &_data[0]);
		}

		constexpr tensor(const useful_specializations::nested_initializer_list_t<T, Rank>& data, const allocator_type& allocator = allocator_type{} ) requires (Rank > 1u)
			: allocator_type { allocator }
		{
			_construct_order_array<Rank>(data);
			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());

			// need to find a way to optimize this in the future
			std::uninitialized_default_construct_n(&_data[0], size_of_current_tensor());
			*this = data; 
		}

		constexpr tensor(const tensor& other)
			: allocator_type { other.get_allocator() }
			, _order_of_dimension(other._order_of_dimension)
			, _size_of_subdimension(other._size_of_subdimension)
			, _data(allocator_type_traits::allocate(get_allocator(), size_of_current_tensor()))
		{
			std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), &_data[0]);
		}

		constexpr tensor(const subdimension<T, Rank>& subdimension, const allocator_type& allocator = allocator_type{})
			: allocator_type { allocator }
			, _data(allocator_type_traits::allocate(get_allocator(), subdimension.size_of_current_tensor()))
		{
			std::copy_n(subdimension._order_of_dimension.begin(), Rank, _order_of_dimension.begin());
			std::copy_n(subdimension._size_of_subdimension.begin(), Rank, _size_of_subdimension.begin());
			std::uninitialized_copy_n(subdimension.cbegin(), size_of_current_tensor(), &_data[0]);
		}

		template<typename ... Args> requires (sizeof...(Args) > Rank && !(is_tensor<Args, T, Rank - 1, allocator_type> && ...))
		constexpr tensor(const Args& ... args) 
		{
			_construct_order_array_and_forward_rest<Rank, Args...>(args...);
		}

		template<typename First, typename ... Args> requires (is_tensor<First, T, Rank - 1, allocator_type>) && (is_tensor<Args, T, Rank - 1, allocator_type> && ...) && (sizeof...(Args) > 0)
		constexpr tensor(const First& first, const Args& ... tensors)
		{
			if (!_are_same_size(first, tensors...))
			{
				throw std::runtime_error("Can't constructor tensor from tensors of different sizes!");
			}

			_order_of_dimension[0] = sizeof...(Args) + 1;
			std::copy_n(first.get_ranks().begin(), Rank - 1, &_order_of_dimension[1]);

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			
			_assign_subdimensions<0, First, Args...>(first, tensors...);
		}

		constexpr auto& operator= (const tensor& other)
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
		constexpr auto& operator= (const Tensor_Type& other) requires (is_tensor<Tensor_Type, T, Rank, allocator_type> && !std::is_same_v<Tensor_Type, tensor>)
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

		constexpr auto& operator= (tensor&& other) 
		{
			if (this != std::addressof(other))
			{
				if constexpr (!std::is_fundamental_v<T>)
				{
					std::destroy_n(_data, size_of_current_tensor());
				}

				allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());

				std::copy_n(other._order_of_dimension.cbegin(), Rank, _order_of_dimension.begin());
				std::copy_n(other._size_of_subdimension.cbegin(), Rank, _size_of_subdimension.begin());
				_data = other._data;

				std::fill_n(other._order_of_dimension.begin(), Rank, 1u);
				std::fill_n(other._size_of_subdimension.begin(), Rank, 1u);

				other._data = allocator_type_traits::allocate(other.get_allocator(), 1u);

				std::uninitialized_default_construct_n(&other._data[0], 1u);
			}
			return *this;
		}

		template<typename Tensor_Type> requires (is_tensor<Tensor_Type, T, Rank, allocator_type>)
		constexpr auto& replace(const Tensor_Type& other) noexcept(TENSORLIB_RELEASE && no_throw_copyable)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (!std::equal(_order_of_dimension.begin(), _order_of_dimension.end(), other._order_of_dimension.begin(), other._order_of_dimension.end()))
				{
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");
				}
			}

			std::copy_n(&other._data[0], size_of_current_tensor(), &_data[0]);

			return *this;
		}

		template <typename First, typename ... Args> requires (is_tensor<First, T, Rank - 1, allocator_type>) && (is_tensor<Args, T, Rank - 1, allocator_type> && ...) && (sizeof...(Args) > 0)
		constexpr auto& replace(const First& first, const Args& ... tensors)
		{
			if (!_are_same_size(first, tensors...))
			{
				throw std::runtime_error("Can't constructor tensor from tensors of different sizes!");
			}

			if (!std::is_fundamental_v<T>)
			{
				std::destroy_n(_data, size_of_current_tensor());
			}

			allocator_type_traits::deallocate(get_allocator(), _data, size_of_current_tensor());

			_order_of_dimension[0] = sizeof...(Args) + 1;
			std::copy_n(first.get_ranks().begin(), Rank - 1, &_order_of_dimension[1]);

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());

			_assign_subdimensions<0, First, Args...>(first, tensors...);

			return *this;
		}

		template<typename Iterator> requires(std::forward_iterator<Iterator> && std::is_convertible_v<typename std::iterator_traits<Iterator>::value_type, T>)
		constexpr auto& replace(Iterator first, Iterator last)
		{
			const auto size = static_cast<size_t>(std::distance(first, last));

			if (size != size_of_current_tensor())
				throw std::runtime_error("Range different in size than tensor!");

			std::copy_n(first, size, begin());

			return (*this);
		}

		constexpr auto& operator=(const useful_specializations::nested_initializer_list_t<T, Rank>& data) requires (Rank > 2u)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (order_of_current_dimension() != data.size())
				{
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
				}
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		constexpr auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) noexcept(TENSORLIB_RELEASE) requires (Rank == 2)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (order_of_current_dimension() != data.size())
				{
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
				}
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		constexpr auto& operator=(const std::initializer_list<T>& data) noexcept(TENSORLIB_RELEASE)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (size_of_current_tensor() != data.size())
				{
					throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");
				}
			}

			std::copy_n(data.begin(), data.size(), begin());

			return (*this);
		}

		template<typename... Sizes> requires (sizeof...(Sizes) == Rank) && useful_concepts::integrals<Sizes>
		constexpr void resize(const Sizes ... new_sizes) 
		{
			const auto old_size = size_of_current_tensor();
			_order_of_dimension = { static_cast<size_t>(new_sizes)... };

			if (TENSORLIB_DEBUGGING)
			{
				if (std::find(_order_of_dimension.cbegin(), _order_of_dimension.cend(), 0u) != _order_of_dimension.cend())
				{
					throw std::runtime_error("Size of dimension can't be changed to zero!");
				}
			}

			std::partial_sum(_order_of_dimension.crbegin(), _order_of_dimension.crend(), _size_of_subdimension.rbegin(), std::multiplies<size_t>());

			std::destroy_n(_data, old_size);
			allocator_type_traits::deallocate(get_allocator(), _data, old_size);

			_data = allocator_type_traits::allocate(get_allocator(), size_of_current_tensor());
			std::uninitialized_default_construct_n(&_data[0], size_of_current_tensor());
		}

		constexpr auto operator[] (const size_t index) noexcept requires (Rank > 1u)
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

		constexpr auto operator[] (const size_t index) const noexcept requires (Rank > 1u)
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

		constexpr auto& operator[] (const size_t index) noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		constexpr auto& operator[] (const size_t index) const noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		constexpr iterator begin() noexcept
		{
			return iterator(&_data[0]);
		}

		constexpr const_iterator begin() const noexcept
		{
			return const_iterator(&_data[0]);
		}

		constexpr const_iterator cbegin() const noexcept
		{
			return const_iterator(&_data[0]);
		}

		constexpr iterator end() noexcept
		{
			return iterator(&_data[size_of_current_tensor()]);
		}

		constexpr const_iterator end() const noexcept
		{
			return const_iterator(&_data[size_of_current_tensor()]);
		}

		constexpr const_iterator cend() const noexcept
		{
			return const_iterator(&_data[size_of_current_tensor()]);
		}

		constexpr auto get_sizes() const noexcept
		{
			return std::span<const size_t, Rank>(_size_of_subdimension);
		}

		constexpr auto get_ranks() const noexcept
		{
			return std::span<const size_t, Rank>(_order_of_dimension);
		}

		constexpr size_t order_of_dimension(const size_t& index) const noexcept
		{
			return _order_of_dimension[index];
		}

		constexpr size_t size_of_subdimension(const size_t& index) const noexcept
		{
			return _size_of_subdimension[index];
		}

		constexpr size_t order_of_current_dimension() const noexcept
		{
			return _order_of_dimension[0];
		}

		constexpr size_t size_of_current_tensor() const noexcept
		{
			return _size_of_subdimension[0];
		}

		constexpr T* data() const noexcept
		{
			return _data;
		}

		constexpr ~tensor()
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
		ConstSourceSizeOfDimensionArraySpan              _order_of_dimension;
		ConstSourceSizeOfSubdimensionArraySpan           _size_of_subdimension;
		ConstSourceDataSpan                              _data;

		static constexpr bool no_throw_default_construction = std::is_nothrow_default_constructible_v<T>;
		static constexpr bool no_throw_destructible = std::is_nothrow_destructible_v<T>;
		static constexpr bool no_throw_copyable = std::is_nothrow_copy_assignable_v<T>;

	public:

		friend class subdimension<T, Rank, allocator_type>;
		friend class tensor<T, Rank, allocator_type>;

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		const_subdimension() = delete;
		const_subdimension(const_subdimension&&) noexcept = default;
		const_subdimension(const const_subdimension&) noexcept = default;

		const_subdimension(const subdimension<T, Rank, allocator_type>& other) noexcept :
			_order_of_dimension{ other._order_of_dimension },
			_size_of_subdimension{ other._size_of_subdimension },
			_data{ other._data }
		{

		}

		template<typename T1, typename T2, typename T3>
		requires std::constructible_from<decltype(_order_of_dimension), T1>
			&& std::constructible_from<decltype(_size_of_subdimension), T2>
			&& std::constructible_from<decltype(_data), T3>
			const_subdimension(T1&& param_1, T2&& param_2, T3&& param_3) noexcept
			: _order_of_dimension(std::forward<T1>(param_1))
			, _size_of_subdimension(std::forward<T2>(param_2))
			, _data(std::forward<T3>(param_3))
		{

		}

		template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
		requires std::constructible_from<decltype(_order_of_dimension), T1, T2>
			&& std::constructible_from<decltype(_size_of_subdimension), T3, T4>
			&& std::constructible_from<decltype(_data), T5, T6>
			const_subdimension(T1&& param_1, T2&& param_2, T3&& param_3, T4&& param_4, T5&& param_5, T6&& param_6) noexcept
			: _order_of_dimension(std::forward<T1>(param_1), std::forward<T2>(param_2))
			, _size_of_subdimension(std::forward<T3>(param_3), std::forward<T4>(param_4))
			, _data(std::forward<T5>(param_5), std::forward<T6>(param_6))
		{

		}

		template<typename Tensor>
		requires std::is_same_v<Tensor, tensor<const T, Rank, allocator_type>>
		const_subdimension(const Tensor& tsor) noexcept :
			_order_of_dimension{ tsor._order_of_dimension.begin(), tsor._order_of_dimension.end() },
			_size_of_subdimension{ tsor._size_of_subdimension.begin(), tsor._size_of_subdimension.end() },
			_data{ tsor.begin(), tsor.end() }
		{

		}

		auto& operator=(const const_subdimension& other) noexcept
		{
			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;
			_data = other._data;

			return *this;
		}

		auto operator[] (const size_t index) const noexcept requires (Rank > 1u)
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

		const T& operator[] (const size_t index) const noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		auto begin() const noexcept
		{
			return const_iterator(_data.data());
		}

		auto end() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		auto cbegin() const noexcept
		{
			return const_iterator(_data.data());
		}

		auto cend() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		auto rank() const noexcept
		{
			return _order_of_dimension;
		}

		size_t order_of_dimension(const size_t& index) const noexcept
		{
			return _order_of_dimension[index];
		}

		size_t size_of_subdimension(const size_t& index) const noexcept
		{
			return _size_of_subdimension[index];
		}

		size_t order_of_current_dimension() const noexcept
		{
			return _order_of_dimension[0];
		}

		size_t size_of_current_tensor() const noexcept
		{
			return _size_of_subdimension[0];
		}

		constexpr auto get_sizes() const noexcept
		{
			return std::span<const size_t, Rank>(_size_of_subdimension);
		}

		constexpr auto get_ranks() const noexcept
		{
			return std::span<const size_t, Rank>(_order_of_dimension);
		}

		const T* data() const noexcept
		{
			return std::to_address(begin());
		}

		bool is_matrix() const noexcept
		{
			return (Rank == 2);
		}

		bool is_square_matrix() const noexcept requires (Rank == 2u)
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

		friend void swap<T, Rank, allocator_type>(subdimension&, subdimension&) noexcept;
		friend void swap<T, Rank, allocator_type>(subdimension&&, subdimension&&) noexcept;

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		subdimension() = delete;
		subdimension(subdimension&&) noexcept = delete;
		subdimension(const subdimension&) noexcept = default;
		subdimension(const const_subdimension<T, Rank>&) noexcept = delete;

		template<typename T1, typename T2, typename T3>
		requires std::constructible_from<decltype(_order_of_dimension), T1>
			&& std::constructible_from<decltype(_size_of_subdimension), T2>
			&& std::constructible_from<decltype(_data), T3>
			subdimension(T1&& param_1, T2&& param_2, T3&& param_3) noexcept
			: _order_of_dimension{ std::forward<T1>(param_1) }
			, _size_of_subdimension{ std::forward<T2>(param_2) }
			, _data{ std::forward<T3>(param_3) }
		{

		}

		template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
		requires std::constructible_from<decltype(_order_of_dimension), T1, T2>
			&& std::constructible_from<decltype(_size_of_subdimension), T3, T4>
			&& std::constructible_from<decltype(_data), T5, T6>
			subdimension(T1&& param_1, T2&& param_2, T3&& param_3, T4&& param_4, T5&& param_5, T6&& param_6) noexcept
			: _order_of_dimension(std::forward<T1>(param_1), std::forward<T2>(param_2))
			, _size_of_subdimension(std::forward<T3>(param_3), std::forward<T4>(param_4))
			, _data(std::forward<T5>(param_5), std::forward<T6>(param_6))
		{

		}

		subdimension(tensor<T, Rank>& mat) noexcept
			: _order_of_dimension{ mat._order_of_dimension.begin(), mat._order_of_dimension.end() }
			, _size_of_subdimension{ mat._size_of_subdimension.begin(), mat._size_of_subdimension.end() }
			, _data{ mat.begin(), mat.end() }
		{

		}

		auto& operator=(const subdimension& other) noexcept
		{
			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;
			_data = other._data;

			return *this;
		}

		template<typename Iterator> requires (std::forward_iterator<Iterator> && std::is_constructible_v<typename std::iterator_traits<Iterator>::value_type, T>)
		auto& replace(Iterator first, Iterator last)
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
		auto& replace(const Tensor_Type& other) noexcept(TENSORLIB_RELEASE)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (!std::equal(_order_of_dimension.begin(), _order_of_dimension.end(), other._order_of_dimension.begin(), other._order_of_dimension.end()))
				{
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");
				}
			}

			std::copy_n(other.cbegin(), size_of_current_tensor(), begin());

			return *this;
		}

		template<typename ... Tensors> requires ((sizeof...(Tensors) > 1) && (is_tensor<Tensors, T, Rank - 1, allocator_type> && ...))
		auto& replace(const Tensors& ... tensors) noexcept(TENSORLIB_RELEASE)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (!_are_same_size((*this)[0], tensors...))
				{
					throw std::runtime_error("Size of tensor we take values from must match the size of current subdimension");
				}
			}

			_assign_subdimensions<0, Tensors...>(tensors...);

			return (*this);
		}

		auto& operator=(const useful_specializations::nested_initializer_list_t<T, Rank>& data) noexcept(TENSORLIB_RELEASE) requires (Rank > 2u)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (order_of_current_dimension() != data.size())
				{
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
				}
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) noexcept(TENSORLIB_RELEASE) requires (Rank == 2u)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (order_of_current_dimension() != data.size())
				{
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");
				}
			}

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		auto& operator=(const std::initializer_list<T>& data) noexcept(TENSORLIB_RELEASE && no_throw_copyable)
		{
			if constexpr (TENSORLIB_DEBUGGING)
			{
				if (size_of_current_tensor() != data.size())
				{
					throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");
				}
			}

			std::copy_n(data.begin(), data.size(), begin());

			return (*this);
		}

		auto& operator=(const tensor<T, Rank, allocator_type>& tensor)
		{
			if (!std::equal(_order_of_dimension.begin(), _order_of_dimension.end(), tensor.get_ranks().begin(), tensor.get_ranks().end()))
			{
				throw std::runtime_error("Size of tensor doesn't match size of subdimension!");
			}

			std::copy_n(tensor.cbegin(), size_of_current_tensor(), begin());

			return (*this);
		}

		auto operator[] (const size_t index) noexcept requires (Rank > 1u)
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

		auto operator[] (const size_t index) const noexcept requires (Rank > 1u)
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

		const T& operator[] (const size_t index) const noexcept requires (Rank == 1u)
		{
			return _data[index];
		}

		T& operator[] (const size_t index) noexcept requires(Rank == 1u)
		{
			return _data[index];
		}

		auto begin() noexcept
		{
			return iterator(_data.data());
		}

		auto begin() const noexcept
		{
			return const_iterator(_data.data());
		}

		auto cbegin() const noexcept
		{
			return const_iterator(_data.data());
		}

		auto end() noexcept
		{
			return iterator(std::to_address(_data.end()));
		}

		auto end() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		auto cend() const noexcept
		{
			return const_iterator(std::to_address(_data.end()));
		}

		size_t& order_of_dimension(const size_t& index) const noexcept
		{
			return _order_of_dimension[index];
		}

		size_t& size_of_subdimension(const size_t& index) const noexcept
		{
			return _size_of_subdimension[index];
		}

		size_t& order_of_current_dimension() const noexcept
		{
			return _order_of_dimension[0];
		}

		size_t& size_of_current_tensor() const noexcept
		{
			return _size_of_subdimension[0];
		}

		T* data() noexcept
		{
			return _data.data();
		}

		constexpr auto get_sizes() const noexcept
		{
			return std::span<const size_t, Rank>(_size_of_subdimension);
		}

		constexpr auto get_ranks() const noexcept
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

		iterator() noexcept : ptr{} {};
		iterator(const iterator&) noexcept = default;
		iterator(iterator&&) noexcept = default;
		iterator(pointer other) noexcept : ptr{ other } {}
		iterator(bool) = delete;

		explicit operator reference() const noexcept
		{
			return *ptr;
		}

		iterator& operator=(const iterator&) = default;

		iterator& operator=(iterator&&) = default;

		iterator& operator= (const pointer other) noexcept
		{
			ptr = other;
			return (*this);
		}

		reference operator* () const noexcept
		{
			return *ptr;
		}

		pointer operator-> () const noexcept
		{
			return ptr;
		}

		iterator& operator++ () noexcept
		{
			++ptr;
			return *this;
		}

		iterator operator++(int) noexcept
		{
			return iterator(ptr++);
		}

		iterator& operator-- () noexcept
		{
			--ptr;
			return *this;
		}

		iterator operator--(int) noexcept
		{
			return iterator(ptr--);
		}

		iterator& operator+=(const ptrdiff_t offset) noexcept
		{
			ptr += offset;
			return *this;
		}

		iterator& operator-=(const ptrdiff_t offset) noexcept
		{
			ptr -= offset;
			return *this;
		}

		reference operator[](const size_t offset) const noexcept
		{
			return ptr[offset];
		}

		friend bool operator==(const iterator it_a, const iterator it_b) noexcept
		{
			return it_a.ptr == it_b.ptr;
		}

		friend bool operator!=(const iterator it_a, const iterator it_b) noexcept
		{
			return it_a.ptr != it_b.ptr;
		}

		friend iterator operator+(const iterator it, const size_t offset) noexcept
		{
			T* result = it.ptr + offset;
			return iterator(result);
		}

		friend iterator operator+(const size_t offset, const iterator& it) noexcept
		{
			auto aux = offset + it.ptr;
			return iterator(aux);
		}

		friend iterator operator-(const iterator it, const size_t offset) noexcept
		{
			T* aux = it.ptr - offset;
			return iterator(aux);
		}

		friend difference_type operator-(const iterator a, const iterator b) noexcept
		{
			return a.ptr - b.ptr;
		}

		friend auto operator<=>(const iterator a, const iterator b) noexcept
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
		using value_type = T;
		using element_type = T;
		using pointer = const T*;
		using reference = const T&;
		using iterator_category = std::contiguous_iterator_tag;

		const_iterator() noexcept : ptr{} {};
		const_iterator(const const_iterator&) noexcept = default;
		const_iterator(const_iterator&&) noexcept = default;
		const_iterator(pointer other) noexcept : ptr{ other } {}

		const_iterator(bool) = delete;

		const_iterator(const iterator other) noexcept : ptr{ other.ptr } {}

		explicit operator reference() const noexcept
		{
			return *ptr;
		}

		const_iterator& operator=(const const_iterator&) noexcept = default;

		const_iterator& operator=(const_iterator&&) noexcept = default;

		const_iterator& operator= (const pointer other) noexcept
		{
			ptr = other;
			return (*this);
		}

		reference operator* () const noexcept
		{
			return *ptr;
		}

		pointer operator-> () const noexcept
		{
			return ptr;
		}

		const_iterator& operator++ () noexcept
		{
			++ptr;
			return *this;
		}

		const_iterator operator++(int) noexcept
		{
			return const_iterator(ptr++);
		}

		const_iterator& operator-- () noexcept
		{
			--ptr;
			return *this;
		}

		const_iterator operator--(int) noexcept
		{
			return const_iterator(ptr--);
		}

		const_iterator& operator+=(const size_t offset) noexcept
		{
			ptr += offset;
			return *this;
		}

		const_iterator& operator-=(const size_t offset) noexcept
		{
			ptr -= offset;
			return *this;
		}

		reference operator[](const size_t offset) const noexcept
		{
			return ptr[offset];
		}

		friend bool operator==(const const_iterator it_a, const const_iterator it_b) noexcept
		{
			return it_a.ptr == it_b.ptr;
		}

		friend bool operator==(const const_iterator it, const pointer adr) noexcept
		{
			return it.ptr == adr;
		}

		friend bool operator!=(const const_iterator it_a, const const_iterator it_b) noexcept
		{
			return it_a.ptr != it_b.ptr;
		}

		friend bool operator!=(const const_iterator it, const pointer adr) noexcept
		{
			return it.ptr != adr;
		}

		friend const_iterator operator+(const const_iterator& it, const size_t offset) noexcept
		{
			return const_iterator(it.ptr + offset);
		}

		friend const_iterator operator+(const size_t offset, const const_iterator it) noexcept
		{
			return const_iterator(offset + it.ptr);
		}

		friend const_iterator operator-(const const_iterator it, const size_t offset) noexcept
		{
			return const_iterator(it.ptr - offset);
		}

		friend difference_type operator-(const const_iterator a, const const_iterator b) noexcept
		{
			return a.ptr - b.ptr;
		}

		friend auto operator<=>(const const_iterator a, const const_iterator b) noexcept
		{
			return a.ptr <=> b.ptr;
		}

	private:

		pointer ptr;
	};

	template <typename T, size_t Rank, typename allocator_type>
	void swap(tensor<T, Rank, allocator_type>& left, tensor<T, Rank, allocator_type>& right) noexcept
	{
		std::swap(left._order_of_dimension, right._order_of_dimension);
		std::swap(left._size_of_subdimension, right._size_of_subdimension);
		std::swap(left._data, right._data);
	}

	template <typename T, size_t Rank, typename allocator_type>
	void swap(subdimension<T, Rank, allocator_type>& left, subdimension<T, Rank, allocator_type>& right) noexcept
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
	void swap(tensor<T, Rank, allocator_type>&& left, tensor<T, Rank, allocator_type>&& right) noexcept
	{
		swap(left, right);
	}

	template <typename T, size_t Rank, typename allocator_type>
	void swap(subdimension<T, Rank, allocator_type>&& left, subdimension<T, Rank, allocator_type>&& right) noexcept
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