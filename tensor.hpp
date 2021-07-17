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

namespace tensor_lib
{
#ifdef _MSC_VER
#ifdef _DEBUG
	constexpr static auto TENSORLIB_DEBUGGING = true;
#else
	constexpr static auto TENSORLIB_DEBUGGING = false;
#endif // _DEBUG
#else
	constexpr static auto TENSORLIB_DEBUGGING = true;
#endif

#define TENSORLIB_NOEXCEPT_IN_RELEASE noexcept(!TENSORLIB_DEBUGGING)

	template <typename T>
	class _tensor_common
	{
	public:
		struct iterator;
		struct const_iterator;
	};

	template <typename T, size_t Rank>
	requires useful_concepts::is_not_zero<size_t, Rank>
		class tensor;

	template <typename T, size_t Rank>
	requires useful_concepts::is_not_zero<size_t, Rank>
		class subdimension;

	template <typename T, size_t Rank>
	requires useful_concepts::is_not_zero<size_t, Rank>
		class const_subdimension;

	template <typename T, typename U, std::size_t Rank>
	requires std::convertible_to<T, U>&& std::convertible_to<U, T>
		void swap(subdimension<T, Rank>&& left, subdimension<U, Rank>&& right);

	template <typename T, typename U, std::size_t Rank>
	requires std::convertible_to<T, U>&& std::convertible_to<U, T>
		void swap(subdimension<T, Rank>& left, subdimension<U, Rank>& right);

	template <typename T, typename U, std::size_t Rank>
	requires std::convertible_to<T, U>&& std::convertible_to<U, T>
		void swap(tensor<T, Rank>& left, tensor<U, Rank>& right);

	template <typename T, size_t Rank>
	requires useful_concepts::is_not_zero<size_t, Rank>
		class tensor : public _tensor_common<T>
	{
	private:
		// Stores the size of each individual dimension of the tensor.
		// Ex: Consider tensor_3d<T>. If _order_of_dimension contains { 3u, 4u, 5u }, it means ours is a tensor of 3x4x5 with a total of 120 elements.
		//
		std::array<size_t, Rank> _order_of_dimension;

		// This is an optimization. Computed when the object is initialized, it contains the equivalent size for each subdimension.
		// Ex: Consider tensor_3d<T>. If _order_of_dimension contains { 3u, 4u, 5u }, 
		// _size_of_subdimension should contain { 120u, 20u, 5u } or { 3u*4u*5u, 4u*5u, 5u }.
		// 
		// This allows methods that rely on the size of our tensor (like "size()") to be O(1) and not have to call std::accumulate() on _order_of_dimension,
		// each time we need the size of a certain dimension.
		//
		std::array<size_t, Rank> _size_of_subdimension;

		// Dynamically allocated data buffer.
		//
		std::unique_ptr<T[]> _data;

	private:
		// Computes _size_of_subdimension at initialization. 
		//
		void construct_size_of_subdimension_array() noexcept
		{
			std::size_t index = Rank - 1;

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

		template <std::size_t Rank_index>
		requires useful_concepts::is_greater_than<std::size_t, std::size_t, Rank_index, 2u>
			void construct_order_array(const useful_specializations::nested_initializer_list<T, Rank_index>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				construct_order_array<Rank_index - 1>(init_list);
			}
		}

		template <std::size_t Rank_index>
		requires useful_concepts::is_equal_to<std::size_t, std::size_t, Rank_index, 1u>
			void construct_order_array(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");

			_order_of_dimension[Rank - Rank_index] = data_size;
		}

		template <std::size_t Rank_index>
		requires useful_concepts::is_equal_to<std::size_t, std::size_t, Rank_index, 2u>
			void construct_order_array(const std::initializer_list<std::initializer_list<T>>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			const auto data_size = data.size();

			if constexpr (TENSORLIB_DEBUGGING)
				if (_order_of_dimension[Rank - Rank_index] != 0 && _order_of_dimension[Rank - Rank_index] != data_size)
					throw std::runtime_error("Initializer list constains uneven number of values for dimensions of equal rank!");

			_order_of_dimension[Rank - Rank_index] = data_size;

			for (const auto& init_list : data)
			{
				construct_order_array<Rank_index - 1>(init_list);
			}
		}

	public:

		friend class subdimension<T, Rank>;
		friend class const_subdimension<T, Rank>;

		template <typename TT, typename U, std::size_t Rank>
		requires std::convertible_to<TT, U>&& std::convertible_to<U, TT>
			friend void swap(tensor<TT, Rank>& left, tensor<U, Rank>& right);

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		// All tensor() constructors take a series of unsigned, non-zero, integeres that reprezent
		// the sizes of each dimension, be it as an array or initializer_list.
		//

		constexpr tensor() noexcept
			: _data(new T[1])
		{
			std::fill(_order_of_dimension.begin(), _order_of_dimension.end(), 1u);
			std::fill(_size_of_subdimension.begin(), _size_of_subdimension.end(), 1u);
		}

		template<typename... Sizes>
		requires useful_concepts::size_of_parameter_pack_equals<Rank, Sizes...>&&
			useful_concepts::constructable_from_common_type<std::size_t, Sizes...>
			tensor(const Sizes ... sizes) noexcept
			: _order_of_dimension{ static_cast<std::size_t>(sizes)... }
		{
			construct_size_of_subdimension_array();
			_data.reset(new T[_size_of_subdimension[0]]);
		}

		tensor(tensor&& other) noexcept
			: _order_of_dimension(std::move(other._order_of_dimension))
			, _size_of_subdimension(std::move(other._size_of_subdimension))
			, _data(std::move(other._data))
		{
			std::fill(other._order_of_dimension.begin(), other._order_of_dimension.end(), 1u);
			std::fill(other._size_of_subdimension.begin(), other._size_of_subdimension.end(), 1u);
			other._data = std::make_unique<T[]>(1);
		}

		tensor(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
		{
			auto data_size = data.size();

			_order_of_dimension[0] = data_size;
			_size_of_subdimension[0] = data_size;

			_data.reset(new T[data_size]);

			std::copy_n(data.begin(), data_size, _data.get());
		}

		tensor(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
		{
			construct_order_array<Rank>(data);
			construct_size_of_subdimension_array();

			_data.reset(new T[_size_of_subdimension[0]]);

			*this = data;
		}

		tensor(const tensor& other) noexcept
			: _order_of_dimension(other._order_of_dimension)
			, _size_of_subdimension(other._size_of_subdimension)
			, _data(new T[other._size_of_subdimension[0]])
		{
			for (std::size_t i = 0; i < _size_of_subdimension[0]; ++i)
			{
				_data[i] = other._data[i];
			}
		}

		tensor(const subdimension<T, Rank>& subdimension) noexcept
			: _data(new T[subdimension._size_of_subdimension[0]])
		{
			std::copy_n(subdimension._order_of_dimension.begin(), Rank, _order_of_dimension.begin());
			std::copy_n(subdimension._size_of_subdimension.begin(), Rank, _size_of_subdimension.begin());
			std::copy_n(subdimension._data.begin(), _size_of_subdimension[0], _data.get());
		}

		auto& operator = (const tensor& other) noexcept
		{
			_order_of_dimension = other._order_of_dimension;
			_size_of_subdimension = other._size_of_subdimension;

			_data.reset(_size_of_subdimension[0]);

			std::copy_n(other._data.get(), _size_of_subdimension[0], _data.get());

			return *this;
		}

		auto& operator = (tensor&& other) noexcept
		{
			if (this != std::addressof(other))
			{
				std::copy_n(other._order_of_dimension.cbegin(), Rank, _order_of_dimension.begin());
				std::copy_n(other._size_of_subdimension.cbegin(), Rank, _size_of_subdimension.begin());
				_data = move(other._data);

				std::fill(other._order_of_dimension.begin(), other._order_of_dimension.end(), 1u);
				std::fill(other._size_of_subdimension.begin(), other._size_of_subdimension.end(), 1u);
				other._data.reset(new T[1]);
			}
			return *this;
		}

		auto& replace(const tensor& other) TENSORLIB_NOEXCEPT_IN_RELEASE
		{
			if constexpr (TENSORLIB_DEBUGGING)
				if (!std::equal(this->_order_of_dimension.begin(), this->_order_of_dimension.end(), other._order_of_dimension.begin()))
					throw std::runtime_error("Size of tensor we take values from must match the size of current tensor");

			for (std::size_t i = 0; i < size_of_current_tensor(); i++)
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

			for (std::size_t i = 0; i < size_of_current_tensor(); i++)
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

			for (std::size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}

			return *this;
		}

		auto& operator=(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
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
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
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
			useful_concepts::constructable_from_common_type<std::size_t, Sizes...>
		{
			_order_of_dimension = { static_cast<std::size_t>(new_sizes)... };

			if constexpr (TENSORLIB_DEBUGGING)
				if (std::find(_order_of_dimension.cbegin(), _order_of_dimension.cend(), 0u) != _order_of_dimension.cend())
					throw std::runtime_error("Size of dimension can't be changed to zero!");

			construct_size_of_subdimension_array();

			_data.reset(new T[size_of_current_tensor()]);
		}

		auto operator[] (const size_t index) noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
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

		auto operator[] (const size_t index) const noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
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

		auto& operator[] (const size_t index) noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
		{
			return _data[index];
		}

		auto& operator[] (const size_t index) const noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
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

		const std::array<std::size_t, Rank>& get_sizes() const noexcept
		{
			return _size_of_subdimension;
		}

		const std::array<std::size_t, Rank>& get_ranks() const noexcept
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
			return _data.get();
		}
	};

	template <typename T, size_t Rank>
	requires useful_concepts::is_not_zero<size_t, Rank>
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

		friend class subdimension<T, Rank>;
		friend class tensor<T, Rank>;

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
		const_subdimension(T1&& param_1, T2&& param_2, T3&& param_3) noexcept
			: _order_of_dimension(std::forward<T1>(param_1))
			, _size_of_subdimension(std::forward<T2>(param_2))
			, _data(std::forward<T3>(param_3))
		{

		}

		template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
		const_subdimension(T1&& param_1, T2&& param_2, T3&& param_3, T4&& param_4, T5&& param_5, T6&& param_6) noexcept
			: _order_of_dimension(std::forward<T1>(param_1), std::forward<T2>(param_2))
			, _size_of_subdimension(std::forward<T3>(param_3), std::forward<T4>(param_4))
			, _data(std::forward<T5>(param_5), std::forward<T6>(param_6))
		{

		}

		const_subdimension(const tensor<const T, Rank>& tsor) noexcept :
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
			requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
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
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
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

		auto get_Rank() const noexcept
		{
			return _order_of_dimension;
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

		T* data() const noexcept
		{
			auto it = _data.begin();
			return reinterpret_cast<const T*>(&it);
		}

		bool is_matrix() const noexcept
		{
			return (Rank == 2);
		}

		bool is_square_matrix() const noexcept
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
		{
			return (_size_of_subdimension[0] == _size_of_subdimension[1]);
		}
	};

	template <typename T, std::size_t Rank>
	requires useful_concepts::is_not_zero<size_t, Rank>
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
		friend class const_subdimension<T, Rank>;
		friend class tensor<T, Rank>;
		friend class tensor<T, Rank + 1>;
		friend class subdimension<T, Rank + 1>;
		friend class tensor<T, useful_specializations::no_zero(Rank - 1)>;

		template <typename TT, typename U, std::size_t RankS>
		requires std::convertible_to<TT, U>&& std::convertible_to<U, TT>
			friend void swap(subdimension<TT, RankS>& left, subdimension<U, RankS>& right);

		template <typename TT, typename U, std::size_t RankS>
		requires std::convertible_to<TT, U>&& std::convertible_to<U, TT>
			friend void swap(subdimension<TT, RankS>&& left, subdimension<U, RankS>&& right);

		using iterator = typename _tensor_common<T>::iterator;
		using const_iterator = typename _tensor_common<T>::const_iterator;

		subdimension() = delete;
		subdimension(subdimension&&) noexcept = delete;
		subdimension(const subdimension&) noexcept = default;
		subdimension(const const_subdimension<T, Rank>&) noexcept = delete;

		template<typename T1, typename T2, typename T3>
		subdimension(T1&& param_1, T2&& param_2, T3&& param_3) noexcept
			: _order_of_dimension{ std::forward<T1>(param_1) }
			, _size_of_subdimension{ std::forward<T2>(param_2) }
			, _data{ std::forward<T3>(param_3) }
		{

		}

		template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
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

			for (std::size_t i = 0; i < size_of_current_tensor(); i++)
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

			for (std::size_t i = 0; i < size_of_current_tensor(); i++)
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

			for (std::size_t i = 0; i < size_of_current_tensor(); i++)
			{
				_data[i] = other._data[i];
			}

			return *this;
		}

		auto& operator=(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE
			requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
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
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
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
			requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
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
			requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
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
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
		{
			return _data[index];
		}

		T& operator[] (const size_t index) noexcept
			requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
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

		friend  bool operator==(const const_iterator it_a, const const_iterator it_b) noexcept
		{
			return it_a.ptr == it_b.ptr;
		}

		friend  bool operator==(const const_iterator it, const pointer adr) noexcept
		{
			return it.ptr == adr;
		}

		friend  bool operator!=(const const_iterator it_a, const const_iterator it_b) noexcept
		{
			return it_a.ptr != it_b.ptr;
		}

		friend  bool operator!=(const const_iterator it, const pointer adr) noexcept
		{
			return it.ptr != adr;
		}

		friend  const_iterator operator+(const const_iterator& it, const size_t offset) noexcept
		{
			return const_iterator(it.ptr + offset);
		}

		friend  const_iterator operator+(const size_t offset, const const_iterator it) noexcept
		{
			return const_iterator(offset + it.ptr);
		}

		friend  const_iterator operator-(const const_iterator it, const size_t offset) noexcept
		{
			return const_iterator(it.ptr - offset);
		}

		friend  const_iterator operator-(const size_t offset, const const_iterator& it) noexcept
		{
			return const_iterator(offset - it.ptr);
		}

		friend  difference_type operator-(const const_iterator a, const const_iterator b) noexcept
		{
			return a.ptr - b.ptr;
		}

		friend  auto operator<=>(const const_iterator a, const const_iterator b) noexcept
		{
			return a.ptr <=> b.ptr;
		}

	private:

		pointer ptr;
	};

	template <typename T, typename U, std::size_t Rank>
	requires std::convertible_to<T, U>&& std::convertible_to<U, T>
		void swap(subdimension<T, Rank>& left, subdimension<U, Rank>& right)
	{
		if constexpr (TENSORLIB_DEBUGGING)
			if (!std::equal(left._order_of_dimension.begin(), left._order_of_dimension.end(), right._order_of_dimension.begin()))
				throw std::runtime_error("Can't swap subdimensions of different sizes!");

		if constexpr (std::is_same_v<T, U> == true)
			if (std::addressof(left) == std::addressof(right))
				return;

		const auto size = left.size_of_current_tensor();

		std::unique_ptr<T[]> aux(new T[size]);

		std::memcpy(aux.get(), left._data.data(), sizeof(T) * size);

		if constexpr (std::is_same_v<T, U> != true)
		{
			std::transform(right._data.begin(), right._data.end(), left._data.begin(), [](const U& val) { return static_cast<T>(val); });

			for (std::size_t i = 0; i < size; i++)
			{
				right._data.data()[i] = static_cast<U>(aux[i]);
			}
		}
		else
		{
			std::copy(right._data.begin(), right._data.end(), left._data.begin());

			for (std::size_t i = 0; i < size; i++)
			{
				right._data.data()[i] = aux[i];
			}
		}
	}

	template <typename T, typename U, std::size_t Rank>
	requires std::convertible_to<T, U>&& std::convertible_to<U, T>
		void swap(subdimension<T, Rank>&& left, subdimension<U, Rank>&& right)
	{
		if constexpr (TENSORLIB_DEBUGGING)
			if (!std::equal(left._order_of_dimension.begin(), left._order_of_dimension.end(), right._order_of_dimension.begin()))
				throw std::runtime_error("Can't swap subdimensions of different sizes!");

		if constexpr (std::is_same_v<T, U> == true)
			if (std::addressof(left) == std::addressof(right))
				return;

		const auto size = left.size_of_current_tensor();

		std::unique_ptr<T[]> aux(new T[size]);

		std::memcpy(aux.get(), left._data.data(), sizeof(T) * size);

		if constexpr (std::is_same_v<T, U> != true)
		{
			std::transform(right._data.begin(), right._data.end(), left._data.begin(), [](const U& val) { return static_cast<T>(val); });

			for (std::size_t i = 0; i < size; i++)
			{
				right._data.data()[i] = static_cast<U>(aux[i]);
			}
		}
		else
		{
			std::copy(right._data.begin(), right._data.end(), left._data.begin());

			for (std::size_t i = 0; i < size; i++)
			{
				right._data.data()[i] = aux[i];
			}
		}
	}

	template <typename T, typename U, std::size_t Rank>
	requires std::convertible_to<T, U>&& std::convertible_to<U, T>
		void swap(tensor<T, Rank>& left, tensor<U, Rank>& right)
	{
		std::swap(left._order_of_dimension, right._order_of_dimension);
		std::swap(left._size_of_subdimension, right._size_of_subdimension);

		if constexpr (std::is_same_v<T, U>)
		{
			std::swap(left._data, right._data);
		}
		else
		{
			if constexpr (sizeof(T) > sizeof(U))
			{
				std::unique_ptr<U[]> aux(new U[left._size_of_subdimension[0]]);

				for (std::size_t i = 0; i != left._size_of_subdimension[0]; ++i)
				{
					aux[i] = right._data[i];
					right._data[i] = static_cast<U>(left._data[i]);
					left._data[i] = static_cast<T>(aux[i]);
				}
			}
			else
			{
				std::unique_ptr<T[]> aux(new T[left._size_of_subdimension[0]]);

				for (std::size_t i = 0; i != left._size_of_subdimension[0]; ++i)
				{
					aux[i] = left._data[i];
					left._data[i] = static_cast<T>(right._data[i]);
					right._data[i] = static_cast<U>(aux[i]);
				}
			}
		}
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