#pragma once

#include "useful_concepts.hpp"
#include "useful_specializations.hpp"

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

#define TENSORLIB_NOEXCEPT_IN_RELEASE noexcept(TENSORLIB_RELEASE)

	template <typename T>
	class _tensor_common
	{
	public:
		struct iterator;
		struct const_iterator;
	};

	template <typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	requires useful_concepts::is_not_zero<Rank>
		class tensor;

	template <typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	requires useful_concepts::is_not_zero<Rank>
		class subdimension;

	template <typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	requires useful_concepts::is_not_zero<Rank>
		class const_subdimension;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(tensor<T, Rank, allocator_type>& left, tensor<T, Rank, allocator_type>& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(subdimension<T, Rank, allocator_type>& left, subdimension<T, Rank, allocator_type>& right);

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(tensor<T, Rank, allocator_type>&& left, tensor<T, Rank, allocator_type>&& right) noexcept;

	template<typename T, size_t Rank, typename allocator_type = std::allocator<std::remove_cv_t<T>>>
	void swap(subdimension<T, Rank, allocator_type>&& left, subdimension<T, Rank, allocator_type>&& right);

	template <typename T, size_t Rank, typename allocator_type>
	requires useful_concepts::is_not_zero<Rank>
		class tensor : public _tensor_common<T>
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
		T * _data{};

		[[no_unique_address]] allocator_type allocator_instance{};

		using allocator_type_traits = std::allocator_traits<allocator_type>;

	private:
		// Computes _size_of_subdimension at initialization. 
		//
		inline void _construct_size_of_subdimension_array() noexcept
		{
			size_t index = Rank - 1;

			_size_of_subdimension[Rank - 1] = _order_of_dimension[Rank - 1];

			if constexpr (Rank - 1 != 0)
			{
				index--;

				while (index)
				{
					_size_of_subdimension[index] = _size_of_subdimension[index + 1] * _order_of_dimension[index];
					index--;
				}

				_size_of_subdimension[0] = _size_of_subdimension[1] * _order_of_dimension[0];
			}
		}

		template <size_t Rank_index>
		requires useful_concepts::is_greater_than<Rank_index, 2u>
			inline void _construct_order_array(const useful_specializations::nested_initializer_list<T, Rank_index>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				_construct_order_array<Rank_index - 1>(init_list);
			}
		}

		template <size_t Rank_index>
		requires useful_concepts::is_equal_to<Rank_index, 1u>
			inline void _construct_order_array(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");

			_order_of_dimension[Rank - Rank_index] = data_size;
		}

		template <size_t Rank_index>
		requires useful_concepts::is_equal_to<Rank_index, 2u>
			inline void _construct_order_array(const std::initializer_list<std::initializer_list<T>>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				_construct_order_array<Rank_index - 1>(init_list);
			}
		}

		inline void _construct_all()
		{
			for (size_t i = 0; i != size_of_current_tensor(); ++i)
			{
				std::construct_at(&_data[i]);
			}
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

		tensor() 
			: _order_of_dimension(useful_specializations::value_initialize_array<size_t, Rank>(1u))
			, _size_of_subdimension(useful_specializations::value_initialize_array<size_t, Rank>(1u))
			, _data(allocator_type_traits::allocate(allocator_instance, 1u))
		{ 
			if constexpr (not std::is_fundamental_v<T>)
				std::construct_at(_data);
		}

		template<typename... Sizes>
		requires useful_concepts::size_of_parameter_pack_equals<Rank, Sizes...>
			&& useful_concepts::integrals<Sizes...>
			&& TENSORLIB_RELEASE
			tensor(const Sizes ... sizes) noexcept
			: _order_of_dimension{ static_cast<size_t>(sizes)... }
		{
			_construct_size_of_subdimension_array();

			_data = allocator_type_traits::allocate(allocator_instance, size_of_current_tensor());
			if constexpr (not std::is_fundamental_v<T>)
				_construct_all();
		}

		template<typename... Sizes>
		requires useful_concepts::size_of_parameter_pack_equals<Rank, Sizes...>
			&& useful_concepts::integrals<Sizes...>
			&& TENSORLIB_DEBUGGING
			tensor(const Sizes ... sizes)
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (useful_specializations::contains_zero(sizes...))
					throw std::runtime_error("Size of subdimension cannot be zero!");

			_order_of_dimension = { static_cast<size_t>(sizes)... };
			_construct_size_of_subdimension_array();

			_data = allocator_type_traits::allocate(allocator_instance, size_of_current_tensor());
			if constexpr (not std::is_fundamental_v<T>)
				_construct_all();
		}

		tensor(tensor&& other) noexcept
			: _order_of_dimension(std::move(other._order_of_dimension))
			, _size_of_subdimension(std::move(other._size_of_subdimension))
			, _data(std::move(other._data))
		{
			std::fill_n(other._order_of_dimension.begin(), Rank, 1u);
			std::fill_n(other._size_of_subdimension.begin(), Rank, 1u);

			other._data = allocator_type_traits::allocate(other.allocator_instance, 1u);
			if constexpr (not std::is_fundamental_v<T>)
				std::construct_at(other._data);
		}

		tensor(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_equal_to<Rank, 1>
		{
			auto data_size = data.size();

			_order_of_dimension[0] = data_size;
			_size_of_subdimension[0] = data_size;

			_data = allocator_type_traits::allocate(allocator_instance, data_size);

			size_t index = 0;

			for (const auto& value : data)
			{
				std::construct_at(&_data[index], value);
			}
		}

		tensor(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_greater_than<Rank, 1>
		{
			_construct_order_array<Rank>(data);
			_construct_size_of_subdimension_array();

			//_data.reset(new T[_size_of_subdimension[0]]);
			_data = allocator_type_traits::allocate(allocator_instance, size_of_current_tensor());
			if constexpr (not std::is_fundamental_v<T>)
				_construct_all();

			*this = data;
		}

		tensor(const tensor& other) noexcept
			: _order_of_dimension(other._order_of_dimension)
			, _size_of_subdimension(other._size_of_subdimension)
			, _data(allocator_type_traits::allocate(allocator_instance, size_of_current_tensor()))
		{
			if constexpr (not std::is_fundamental_v<T>)
				std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), &_data[0]);
			else
				std::copy_n(other.cbegin(), size_of_current_tensor(), &_data[0]);
		}

		tensor(const subdimension<T, Rank>& subdimension) noexcept
			: _data(allocator_type_traits::allocate(allocator_instance, subdimension.size_of_current_tensor()))
		{
			std::copy_n(subdimension._order_of_dimension.begin(), Rank, _order_of_dimension.begin());
			std::copy_n(subdimension._size_of_subdimension.begin(), Rank, _size_of_subdimension.begin());
			if constexpr (not std::is_fundamental_v<T>)
				std::uninitialized_copy_n(subdimension.cbegin(), size_of_current_tensor(), &_data[0]);
			else
				std::copy_n(subdimension.cbegin(), size_of_current_tensor(), &_data[0]);
		}

		auto& operator = (const tensor& other) noexcept
		{
			if constexpr (not std::is_fundamental_v<T>)
				std::destroy_n(_data, size_of_current_tensor());
			allocator_type_traits::deallocate(allocator_instance, _data, size_of_current_tensor());

			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;

			_data = allocator_type_traits::allocate(allocator_instance, size_of_current_tensor());
			if constexpr (not std::is_fundamental_v<T>)
				std::uninitialized_copy_n(other.cbegin(), size_of_current_tensor(), &_data[0]);
			else
				std::copy_n(other.cbegin(), size_of_current_tensor(), &_data[0]);

			return *this;
		}

		auto& operator = (tensor&& other) noexcept
		{
			if (this != std::addressof(other))
			{
				if constexpr (not std::is_fundamental_v<T>)
					std::destroy_n(_data, size_of_current_tensor());
				allocator_type_traits::deallocate(allocator_instance, _data, size_of_current_tensor());

				std::copy_n(other._order_of_dimension.cbegin(), Rank, _order_of_dimension.begin());
				std::copy_n(other._size_of_subdimension.cbegin(), Rank, _size_of_subdimension.begin());
				_data = other._data;

				std::fill_n(other._order_of_dimension.begin(), Rank, 1u);
				std::fill_n(other._size_of_subdimension.begin(), Rank, 1u);

				other._data = allocator_type_traits::allocate(other.allocator_instance, 1u);
				
				if constexpr (not std::is_fundamental_v<T>)
					std::construct_at(other._data);
			}
			return *this;
		}

		auto& replace(const tensor& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			/*for (size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}*/
			std::copy_n(other._data, size_of_current_tensor(), _data);

			return *this;
		}

		auto& replace(const subdimension<T, Rank>& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			/*for (size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}*/
			std::copy_n(&other._data[0], size_of_current_tensor(), &_data[0]);

			return *this;
		}

		auto& replace(const const_subdimension<T, Rank>& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			/*for (size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}*/
			std::copy_n(&other._data[0], size_of_current_tensor(), &_data[0]);

			return *this;
		}

		auto& operator=(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_greater_than<Rank, 2>
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (order_of_current_dimension() != data.size())
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_equal_to<Rank, 2>
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (order_of_current_dimension() != data.size())
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		auto& operator=(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (size_of_current_tensor() != data.size())
					throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");

			std::copy_n(data.begin(), data.size(), begin());

			return (*this);
		}

		template<typename... Sizes>
		void resize(const Sizes ... new_sizes) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::size_of_parameter_pack_equals<Rank, Sizes...>&&
			useful_concepts::constructable_from_common_type<size_t, Sizes...>
		{
			const auto old_size = size_of_current_tensor();
			_order_of_dimension = { static_cast<size_t>(new_sizes)... };

			if constexpr (TENSORLIB_DEBUGGING)
				if (std::find(_order_of_dimension.cbegin(), _order_of_dimension.cend(), 0u) != _order_of_dimension.cend())
					throw std::runtime_error("Size of dimension can't be changed to zero!");

			_construct_size_of_subdimension_array();

			if constexpr (not std::is_fundamental_v<T>)
				std::destroy_n(_data, old_size);
			allocator_type_traits::deallocate(allocator_instance, _data, old_size);

			_data = allocator_type_traits::allocate(allocator_instance, size_of_current_tensor());
			if constexpr (not std::is_fundamental_v<T>)
				_construct_all();

			//_data.reset(new T[size_of_current_tensor()]);
		}

		auto operator[] (const size_t index) noexcept requires useful_concepts::is_greater_than<Rank, 1>
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

		auto operator[] (const size_t index) const noexcept requires useful_concepts::is_greater_than<Rank, 1>
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

		auto& operator[] (const size_t index) noexcept requires useful_concepts::is_equal_to<Rank, 1>
		{
			return _data[index];
		}

		auto& operator[] (const size_t index) const noexcept requires useful_concepts::is_equal_to<Rank, 1>
		{
			return _data[index];
		}

		iterator begin() noexcept
		{
			return iterator(&_data[0]);
		}

		const_iterator begin() const noexcept
		{
			return const_iterator(&_data[0]);
		}

		const_iterator cbegin() const noexcept
		{
			return const_iterator(&_data[0]);
		}

		iterator end() noexcept
		{
			return iterator(&_data[size_of_current_tensor()]);
		}

		const_iterator end() const noexcept
		{
			return const_iterator(&_data[size_of_current_tensor()]);
		}

		const_iterator cend() const noexcept
		{
			return const_iterator(&_data[size_of_current_tensor()]);
		}

		const std::array<size_t, Rank>& get_sizes() const noexcept
		{
			return _size_of_subdimension;
		}

		const std::array<size_t, Rank>& get_ranks() const noexcept
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

		T* data() const noexcept
		{
			return _data;
		}

		~tensor()
		{
			if constexpr (not std::is_fundamental_v<T>)
				std::destroy_n(_data, size_of_current_tensor());
			allocator_type_traits::deallocate(allocator_instance, _data, size_of_current_tensor());
		}
	};

	template <typename T, size_t Rank, typename allocator_type>
	requires useful_concepts::is_not_zero<Rank>
		class const_subdimension : public _tensor_common<T>
	{
		using ConstSourceSizeOfDimensionArraySpan = std::span<const size_t, Rank>;
		using ConstSourceSizeOfSubdimensionArraySpan = std::span<const size_t, Rank>;
		using ConstSourceDataSpan = std::span<const T>;

	private:
		ConstSourceSizeOfDimensionArraySpan              _order_of_dimension;
		ConstSourceSizeOfSubdimensionArraySpan           _size_of_subdimension;
		ConstSourceDataSpan                              _data;

	public:

		friend class subdimension<T, Rank, allocator_type>;
		friend class tensor<T, Rank, allocator_type>;

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		const_subdimension() = delete;
		const_subdimension(const_subdimension&&) noexcept = delete;
		const_subdimension(const const_subdimension&) noexcept = default;

		const_subdimension(const subdimension<T, Rank>& other) noexcept :
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

		const_subdimension(const tensor<const T, Rank, allocator_type>& tsor) noexcept :
			_order_of_dimension{ tsor._order_of_dimension.begin(),      tsor._order_of_dimension.end() },
			_size_of_subdimension{ tsor._size_of_subdimension.begin(),    tsor._size_of_subdimension.end() },
			_data{ tsor.begin(),                          tsor.end() }
		{

		}

		auto& operator=(const const_subdimension& other) noexcept
		{
			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;
			_data = other._data;

			return *this;
		}

		auto operator[] (const size_t index) const noexcept
			requires useful_concepts::is_greater_than<Rank, 1>
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

		const T& operator[] (const size_t index) const noexcept
			requires useful_concepts::is_equal_to<Rank, 1>
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

		const T* data() const noexcept
		{
			return std::to_address(begin());
		}

		bool is_matrix() const noexcept
		{
			return (Rank == 2);
		}

		bool is_square_matrix() const noexcept
			requires useful_concepts::is_equal_to<Rank, 2>
		{
			return (_size_of_subdimension[0] == _size_of_subdimension[1]);
		}
	};

	template <typename T, size_t Rank, typename allocator_type>
	requires useful_concepts::is_not_zero<Rank>
		class subdimension : public _tensor_common<T>
	{
		using SourceSizeOfDimensionArraySpan = std::span<size_t, Rank>;
		using SourceSizeOfSubdimensionArraySpan = std::span<size_t, Rank>;
		using SourceDataSpan = std::span<T>;

	private:
		SourceSizeOfDimensionArraySpan              _order_of_dimension;
		SourceSizeOfSubdimensionArraySpan           _size_of_subdimension;
		SourceDataSpan                              _data;

	public:
		friend class const_subdimension<T, Rank, allocator_type>;
		friend class tensor<T, Rank, allocator_type>;
		friend class tensor<T, Rank + 1, allocator_type>;
		friend class subdimension<T, Rank + 1, allocator_type>;
		friend class tensor<T, useful_specializations::exclude_zero(Rank - 1), allocator_type>;

		friend void swap<T, Rank, allocator_type>(subdimension&, subdimension&);
		friend void swap<T, Rank, allocator_type>(subdimension&&, subdimension&&);

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
			: _order_of_dimension{ mat._order_of_dimension.begin(),      mat._order_of_dimension.end() }
			, _size_of_subdimension{ mat._size_of_subdimension.begin(),    mat._size_of_subdimension.end() }
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

		auto& replace(const tensor<T, Rank>& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			for (size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}

			return *this;
		}

		auto& replace(const subdimension<T, Rank>& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			for (size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}

			return *this;
		}

		auto& replace(const const_subdimension<T, Rank>& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			for (size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}

			return *this;
		}

		auto& operator=(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_greater_than<Rank, 2>
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (order_of_current_dimension() != data.size())
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_equal_to<Rank, 2>
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (order_of_current_dimension() != data.size())
					throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

			auto it = data.begin();

			for (size_t index = 0; index < order_of_current_dimension(); index++)
			{
				(*this)[index] = *(it + index);
			}

			return (*this);
		}

		auto& operator=(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (size_of_current_tensor() != data.size())
					throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");

			std::copy_n(data.begin(), data.size(), begin());

			return (*this);
		}

		auto operator[] (const size_t index) noexcept
			requires useful_concepts::is_greater_than<Rank, 1>
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

		auto operator[] (const size_t index) const noexcept
			requires useful_concepts::is_greater_than<Rank, 1>
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

		const T& operator[] (const size_t index) const noexcept
			requires useful_concepts::is_equal_to<Rank, 1>
		{
			return _data[index];
		}

		T& operator[] (const size_t index) noexcept
			requires useful_concepts::is_equal_to<Rank, 1>
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

		auto get_Rank() const noexcept
		{
			std::span<const size_t> dims_sizes{ _order_of_dimension.begin(), _order_of_dimension.end() };

			return dims_sizes;
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
			auto it = _data.begin();
			return reinterpret_cast<T*>(&it);
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

		reference operator* (void) const noexcept
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
			auto aux = ptr;
			ptr++;
			return iterator(aux);
		}

		iterator& operator-- () noexcept
		{
			--ptr;
			return *this;
		}

		iterator operator--(int) noexcept
		{
			auto aux = ptr;
			ptr--;
			return iterator(aux);
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
			return *(ptr + offset);
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

		friend iterator operator-(const size_t offset, const iterator it) noexcept
		{
			auto aux = offset - it.ptr;
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

		reference operator* (void) const noexcept
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
			auto aux = ptr;
			ptr++;
			return const_iterator(aux);
		}

		const_iterator& operator-- () noexcept
		{
			--ptr;
			return *this;
		}

		const_iterator operator--(int) noexcept
		{
			auto aux = ptr;
			ptr--;
			return const_iterator(aux);
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
			return *(ptr + offset);
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

		friend const_iterator operator-(const size_t offset, const const_iterator& it) noexcept
		{
			return const_iterator(offset - it.ptr);
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
		std::swap(left._order_of_dimension,		right._order_of_dimension);
		std::swap(left._size_of_subdimension,	right._size_of_subdimension);
		std::swap(left._data,					right._data); 
	}

	template <typename T, size_t Rank, typename allocator_type>
	void swap(subdimension<T, Rank, allocator_type>& left, subdimension<T, Rank, allocator_type>& right)
	{
		if (!std::equal(left._order_of_dimension.begin(),	left._order_of_dimension.end(),		right._order_of_dimension.begin()))
			throw std::runtime_error("Can't swap subdimensions of different sizes!\n");

		const auto size = left.size_of_current_tensor();
		T* aux = new T[size];

		std::copy(left.cbegin(),	left.cend(),	aux);
		std::copy(right.cbegin(),	right.cend(),	left.begin());
		std::copy(aux,				&aux[size],		right.begin());

		delete[] aux;
	}

	template <typename T, size_t Rank, typename allocator_type>
	void swap(tensor<T, Rank, allocator_type>&& left, tensor<T, Rank, allocator_type>&& right) noexcept
	{
		std::swap(left._order_of_dimension,		right._order_of_dimension);
		std::swap(left._size_of_subdimension,	right._size_of_subdimension);
		std::swap(left._data,					right._data);
	}

	template <typename T, size_t Rank, typename allocator_type>
	void swap(subdimension<T, Rank, allocator_type>&& left, subdimension<T, Rank, allocator_type>&& right)
	{
		if (!std::equal(left._order_of_dimension.begin(),	left._order_of_dimension.end(),		right._order_of_dimension.begin()))
			throw std::runtime_error("Can't swap subdimensions of different sizes!\n");

		const auto size = left.size_of_current_tensor();
		T* aux = new T[size];

		std::copy(left.cbegin(),	left.cend(),	aux);
		std::copy(right.cbegin(),	right.cend(),	left.begin());
		std::copy(aux,				&aux[size],		right.begin());

		delete[] aux;
	}

	namespace aliases
	{
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
}