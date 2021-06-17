#pragma once

#include <array>
#include <numeric>
#include <utility>
#include <span>
#include "useful_concepts.hpp"
#include "useful_specializations.hpp"

namespace tensor_lib
{
    template <typename T, size_t Rank>
    class subdimension;

    template <typename T, size_t Rank>
    class const_subdimension;

	template <typename T, size_t Rank> requires useful_concepts::is_greater_than<size_t, size_t, Rank, 0>
	class tensor
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


        // Computes _size_of_subdimension at initialization. 
        //
        constexpr void construct_size_of_subdimension_array() noexcept
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

    public:

        friend class subdimension<T, Rank>;
        friend class const_subdimension<T, Rank>;

        struct iterator
        {
            using difference_type = ptrdiff_t;
            using value_type = T;
            using element_type = T;
            using pointer = T*;
            using reference = T&;
            using iterator_category = std::contiguous_iterator_tag;

            constexpr iterator() noexcept : ptr{} {};
            constexpr iterator(const iterator&) noexcept = default;
            constexpr iterator(iterator&&) noexcept = default;
            constexpr iterator(pointer other) noexcept : ptr{ other } {}

            constexpr iterator(bool) = delete;

            constexpr reference get() noexcept
            {
                return *ptr;
            }

            constexpr const reference get() const noexcept
            {
                return *ptr;
            }

            constexpr explicit operator reference() const noexcept
            {
                return *ptr;
            }

            constexpr iterator& operator=(const iterator&) = default;

            constexpr iterator& operator=(iterator&&) = default;

            constexpr iterator& operator= (const pointer other) noexcept
            {
                ptr = other;
                return (*this);
            }

            constexpr reference operator* (void) const noexcept
            {
                return *ptr;
            }

            constexpr pointer operator-> () const noexcept
            {
                return ptr;
            }

            constexpr iterator& operator++ () noexcept
            {
                ++ptr;
                return *this;
            }

            constexpr iterator operator++(int) noexcept
            {
                auto aux = ptr;
                ptr++;
                return iterator(aux);
            }

            constexpr iterator& operator-- () noexcept
            {
                --ptr;
                return *this;
            }

            constexpr iterator operator--(int) noexcept
            {
                auto aux = ptr;
                ptr--;
                return iterator(aux);
            }

            constexpr iterator& operator+=(const size_t& offset) noexcept
            {
                ptr += offset;
                return *this;
            }

            constexpr iterator& operator-=(const size_t& offset) noexcept
            {
                ptr -= offset;
                return *this;
            }

            constexpr reference operator[](const size_t& offset) const noexcept
            {
                return *(ptr + offset);
            }

            friend constexpr bool operator==(const iterator& it_a, const iterator& it_b) noexcept
            {
                return it_a.ptr == it_b.ptr;
            }

            friend constexpr bool operator==(const iterator& it, const pointer adr) noexcept
            {
                return it.ptr == adr;
            }

            friend constexpr bool operator!=(const iterator& it_a, const iterator& it_b) noexcept
            {
                return it_a.ptr != it_b.ptr;
            }

            friend constexpr bool operator!=(const iterator& it, const pointer adr) noexcept
            {
                return it.ptr != adr;
            }

            friend constexpr iterator operator+(const iterator& it, const size_t& offset) noexcept
            {
                T* result = it.ptr + offset;
                return iterator(result);
            }

            friend constexpr iterator operator+(const int& offset, const iterator& it) noexcept
            {
                auto aux = offset + it.ptr;
                return iterator(aux);
            }

            friend constexpr iterator operator-(const iterator& it, const size_t& offset) noexcept
            {
                T* aux = it.ptr - offset;
                return iterator(aux);
            }

            friend constexpr iterator operator-(const int& offset, const iterator& it) noexcept
            {
                auto aux = offset - it.ptr;
                return iterator(aux);
            }

            friend constexpr difference_type operator-(const iterator& a, const iterator& b) noexcept
            {
                return a.ptr - b.ptr;
            }

            friend constexpr auto operator<=>(const iterator& a, const iterator& b) noexcept
            {
                return a.ptr <=> b.ptr;
            }

            friend constexpr auto operator<=>(const iterator& it, const pointer pt) noexcept
            {
                return it.ptr <=> pt;
            }

        private:

            pointer ptr;
        };

        struct const_iterator
        {
            using difference_type = ptrdiff_t;
            using value_type = T;
            using element_type = T;
            using pointer = const T*;
            using reference = const T&;
            using iterator_category = std::contiguous_iterator_tag;

            constexpr const_iterator() noexcept : ptr{} {};
            constexpr const_iterator(const const_iterator&) noexcept = default;
            constexpr const_iterator(const_iterator&&) noexcept = default;
            constexpr const_iterator(pointer other) noexcept : ptr{ other } {}

            constexpr const_iterator(bool) = delete;

            constexpr const_iterator(const iterator& other) noexcept : ptr{ other.ptr } {}

            constexpr reference get() const noexcept
            {
                return *ptr;
            }

            constexpr explicit operator const reference() const noexcept
            {
                return *ptr;
            }

            constexpr const_iterator& operator=(const const_iterator&) noexcept = default;

            constexpr const_iterator& operator=(const_iterator&&) noexcept = default;

            constexpr const_iterator& operator= (const pointer other) noexcept
            {
                ptr = other;
                return (*this);
            }

            constexpr reference operator* (void) const noexcept
            {
                return *ptr;
            }

            constexpr pointer operator-> () const noexcept
            {
                return ptr;
            }

            constexpr const_iterator& operator++ () noexcept
            {
                ++ptr;
                return *this;
            }

            constexpr const_iterator operator++(int) noexcept
            {
                auto aux = ptr;
                ptr++;
                return const_iterator(aux);
            }

            constexpr const_iterator& operator-- () noexcept
            {
                --ptr;
                return *this;
            }

            constexpr const_iterator operator--(int) noexcept
            {
                auto aux = ptr;
                ptr--;
                return const_iterator(aux);
            }

            constexpr const_iterator& operator+=(const size_t& offset) noexcept
            {
                ptr += offset;
                return *this;
            }

            constexpr const_iterator& operator-=(const size_t& offset) noexcept
            {
                ptr -= offset;
                return *this;
            }

            constexpr reference operator[](const size_t& offset) const noexcept
            {
                return *(ptr + offset);
            }

            friend constexpr bool operator==(const const_iterator& it_a, const const_iterator& it_b) noexcept
            {
                return it_a.ptr == it_b.ptr;
            }

            friend constexpr bool operator==(const const_iterator& it, const pointer adr) noexcept
            {
                return it.ptr == adr;
            }

            friend constexpr bool operator!=(const const_iterator& it_a, const const_iterator& it_b) noexcept
            {
                return it_a.ptr != it_b.ptr;
            }

            friend constexpr bool operator!=(const const_iterator& it, const pointer adr) noexcept
            {
                return it.ptr != adr;
            }

            friend constexpr const_iterator operator+(const const_iterator& it, const size_t& offset) noexcept
            {
                pointer result = it.ptr + offset;
                return const_iterator(result);
            }

            friend constexpr const_iterator operator+(const int& offset, const const_iterator& it) noexcept
            {
                auto aux = offset + it.ptr;
                return const_iterator(aux);
            }

            friend constexpr const_iterator operator-(const const_iterator& it, const size_t& offset) noexcept
            {
                T* aux = it.ptr - offset;
                return const_iterator(aux);
            }

            friend constexpr const_iterator operator-(const int& offset, const const_iterator& it) noexcept
            {
                auto aux = offset - it.ptr;
                return const_iterator(aux);
            }

            friend constexpr difference_type operator-(const const_iterator& a, const const_iterator& b) noexcept
            {
                return a.ptr - b.ptr;
            }

            friend constexpr auto operator<=>(const const_iterator& a, const const_iterator& b) noexcept
            {
                return a.ptr <=> b.ptr;
            }

            friend constexpr auto operator<=>(const const_iterator& it, const pointer pt) noexcept
            {
                return it.ptr <=> pt;
            }

        private:

            pointer ptr;
        };

        // All tensor() constructors take a series of unsigned, non-zero, integeres that reprezent
        // the sizes of each dimension, be it as an array or initializer_list.
        //

		constexpr tensor(const std::array<size_t, Rank>& sizes)
		{
			for (const auto& value : sizes)
				if (value == 0)
					throw std::runtime_error("Can't initialize tensor with a dimension of size zero!");

            _order_of_dimension = sizes;

            construct_size_of_subdimension_array();

            _data.reset(new T[_size_of_subdimension[0]]);
		}

        constexpr tensor(const std::initializer_list<size_t>& sizes) 
		{
			if (sizes.size() != Rank)
				throw std::runtime_error("The number of sizes given doesn't match the number of Rank of the tensor!");
			for (const auto& value : sizes)
				if (value == 0)
					throw std::runtime_error("Can't initialize tensor with a dimension of size zero!");

			std::copy(sizes.begin(), sizes.end(), _order_of_dimension.begin());

            construct_size_of_subdimension_array();

            _data.reset(new T[_size_of_subdimension[0]]);
		}

        template<typename... Sizes> requires useful_concepts::size_of_parameter_pack_equals<Sizes..., Rank>
        constexpr tensor(Sizes ... sizes) : _order_of_dimension{ sizes... }
        {
            construct_size_of_subdimension_array();
            _data.reset(new T[_size_of_subdimension[0]]);
        }

        constexpr tensor(tensor&& other) noexcept
        {
            std::copy(other._order_of_dimension.begin(), other._order_of_dimension.end(), _order_of_dimension.begin());
            std::copy(other._size_of_subdimension.begin(), other._size_of_subdimension.end(), _size_of_subdimension.begin());

            _data = std::move(other._data);
            _data.reset(new T[_size_of_subdimension[0]]);
        }

        constexpr tensor(const tensor& other)
        {
            std::copy(other._order_of_dimension.begin(), other._order_of_dimension.end(), _order_of_dimension.begin());
            std::copy(other._size_of_subdimension.begin(), other._size_of_subdimension.end(), _size_of_subdimension.begin());

            _data.reset(new T[_size_of_subdimension[0]]);
            
            for (size_t i = 0; i < _size_of_subdimension[0]; i++)
            {
                _data[i] = other._data[i];
            }
        }

        constexpr tensor(const subdimension<T, Rank>& subdimension)
        {
            std::copy(subdimension._order_of_dimension.begin(), subdimension._order_of_dimension.end(), _order_of_dimension.begin());
            std::copy(subdimension._size_of_subdimension.begin(), subdimension._size_of_subdimension.end(), _size_of_subdimension.begin());

            _data.reset(new T[_size_of_subdimension[0]]);

            for (size_t i = 0; i < _size_of_subdimension[0]; i++)
            {
                _data[i] = subdimension._data[i];
            }
        }

        constexpr auto operator = (const tensor& other)
        {
            auto* ptr = _data.release(); 
            _data.get_deleter() (ptr);

            _order_of_dimension = other._order_of_dimension;
            _size_of_subdimension = other._size_of_subdimension;

            _data = std::make_unique<T[]>(_size_of_subdimension[0]);

            for (size_t i = 0; i < _size_of_subdimension[0]; i++)
            {
                _data[i] = other._data[i];
            }

            return (*this);
        }

        constexpr auto operator = (tensor&& other) noexcept
        {
            auto* ptr = _data.release();
            _data.get_deleter() (ptr);

            _order_of_dimension = other._order_of_dimension;
            _size_of_subdimension = other._size_of_subdimension;

            _data = std::move(other._data);
            other._data = std::make_unique<T[]>(other._size_of_subdimension[0]);

            return (*this);
        }

        constexpr auto& operator=(const useful_specializations::nested_initializer_list<T, Rank>& data) requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
        {
            if (order_of_current_dimension() != data.size())
                throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

            auto it = data.begin();

            for (size_t index = 0; index < order_of_current_dimension(); index++)
            {
                (*this)[index] = *(it + index);
            }

            return (*this);
        }

        constexpr auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
        {
            if (order_of_current_dimension() != data.size())
                throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

            auto it = data.begin();

            for (size_t index = 0; index < order_of_current_dimension(); index++)
            {
                (*this)[index] = *(it + index);
            }

            return (*this);
        }

        constexpr auto& operator=(const std::initializer_list<T>& data)
        {
            if (size_of_current_tensor() != data.size())
                throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");

            std::copy(data.begin(), data.end(), begin());

            return (*this);
        }

        constexpr void resize(const std::array<size_t, Rank>& new_sizes)
        {
            for (const auto& val : new_sizes)
                if (val == 0)
                    throw std::runtime_error("Can't have a zero-sized dimension!");

            auto current_size = _size_of_subdimension[0];

            std::copy(new_sizes.begin(), new_sizes.end(), _order_of_dimension.begin());
            construct_size_of_subdimension_array();

            std::unique_ptr<T[]> new_data = std::make_unique<T[]>(_size_of_subdimension[0]);

            auto min = std::min(current_size, _size_of_subdimension[0]);

            for (size_t i = 0; i < min; i++)
                new_data[i] = _data[i];

            delete[] _data;

            _data = std::move(new_data);
        }

        constexpr void resize(const std::span<const size_t>& new_sizes) 
        {
            if (new_sizes.size() != Rank)
                throw std::runtime_error("Number of Rank given doesn't match the number of Rank of the tensor!");
            for (const auto& val : new_sizes)
                if (val == 0)
                    throw std::runtime_error("Can't have a zero-sized dimension!");

            auto current_size = _size_of_subdimension[0];

            //_order_of_dimension = new_sizes;
            std::copy(new_sizes.begin(), new_sizes.end(), _order_of_dimension.begin());
            construct_size_of_subdimension_array();

            std::unique_ptr<T[]> new_data = std::make_unique<T[]>(_size_of_subdimension[0]);

            auto min = std::min(current_size, _size_of_subdimension[0]);

            for (size_t i = 0; i < min; i++)
                new_data[i] = _data[i];

            delete[] _data;

            _data = std::move(new_data);
        }

        constexpr void resize(const std::initializer_list<size_t>& new_sizes)
        {
            if (new_sizes.size() != Rank)
                throw std::runtime_error("Number of Rank given doesn't match the number of Rank of the tensor!");
            for (const auto& val : new_sizes)
                if (val == 0)
                    throw std::runtime_error("Can't have a zero-sized dimension!");

            auto current_size = _size_of_subdimension[0];

            //_order_of_dimension = new_sizes;
            std::copy(new_sizes.begin(), new_sizes.end(), _order_of_dimension.begin());
            construct_size_of_subdimension_array();

            std::unique_ptr<T[]> new_data = std::make_unique<T[]>(_size_of_subdimension[0]);

            auto min = std::min(current_size, _size_of_subdimension[0]);

            for (size_t i = 0; i < min; i++)
                new_data[i] = _data[i];

            delete[] _data;

            _data = std::move(new_data);
        }

        constexpr auto operator[] (const size_t index) noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
        {
            return subdimension<T, Rank - 1>
                (
                    std::span<size_t, Rank - 1>(_order_of_dimension.begin() + 1, _order_of_dimension.begin() + Rank),
                    std::span<size_t, Rank - 1>(_size_of_subdimension.begin() + 1, _size_of_subdimension.begin() + Rank),
                    std::span<T>(begin() + index * _size_of_subdimension[1], begin() + index * _size_of_subdimension[1] + _size_of_subdimension[1])
                );
        }

        constexpr const auto operator[] (const size_t index) const noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
        {
            return const_subdimension<T, Rank - 1>
                (
                    std::span<const size_t, Rank - 1> (_order_of_dimension.cbegin() + 1, _order_of_dimension.cbegin() + Rank),
                    std::span<const size_t, Rank - 1> (_size_of_subdimension.cbegin() + 1, _size_of_subdimension.cbegin() + Rank),
                    std::span<const T>(cbegin() + index * _size_of_subdimension[1], cbegin() + index * _size_of_subdimension[1] + _size_of_subdimension[1])
                );
        }

        constexpr auto& operator[] (const size_t index) noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
        {
            return _data[index];
        }

        constexpr const auto& operator[] (const size_t index) const noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
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
            std::span<const size_t> dims_sizes{ _order_of_dimension.begin(), _order_of_dimension.end() };

            return dims_sizes;
        }

        constexpr const size_t& order_of_dimension(const size_t& index) const noexcept
        {
            return _order_of_dimension[index];
        }

        constexpr const size_t& size_of_subdimension(const size_t& index) const noexcept
        {
            return _size_of_subdimension[index];
        }

        constexpr const size_t& order_of_current_dimension() const noexcept
        {
            return _order_of_dimension[0];
        }

        constexpr const size_t& size_of_current_tensor() const noexcept
        {
            return _size_of_subdimension[0];
        }

        constexpr const T* data() noexcept
        {
            return _data.get();
        }
	};

    template <typename T, size_t Rank>
    class const_subdimension
    {
        using ConstSourceSizeOfDimensionArraySpan        = std::span<const size_t, Rank>;
        using ConstSourceSizeOfSubdimensionArraySpan     = std::span<const size_t, Rank>;
        using ConstSourceDataSpan                        = std::span<const T>;

    private:
        ConstSourceSizeOfDimensionArraySpan              _order_of_dimension;
        ConstSourceSizeOfSubdimensionArraySpan           _size_of_subdimension;
        ConstSourceDataSpan                              _data;

    public:

        friend class subdimension<T, Rank>;
        friend class tensor<T, Rank>;

        constexpr const_subdimension() = delete;
        constexpr const_subdimension(const_subdimension&&) noexcept = delete;
        constexpr const_subdimension(const const_subdimension&) noexcept = default;

        constexpr const_subdimension(const subdimension<T, Rank>& other) noexcept :
            _order_of_dimension{ other._order_of_dimension },
            _size_of_subdimension{ other._size_of_subdimension },
            _data{ other._data }
        {

        }

        constexpr const_subdimension(const ConstSourceSizeOfDimensionArraySpan& size_of_dimension_span, const ConstSourceSizeOfSubdimensionArraySpan& size_of_subdimension_span, const ConstSourceDataSpan& data_span) noexcept :
            _order_of_dimension{ size_of_dimension_span },
            _size_of_subdimension{ size_of_subdimension_span },
            _data{ data_span }
        {

        }

        constexpr const_subdimension(const tensor<const T, Rank>& mat) noexcept :
            _order_of_dimension{ mat._order_of_dimension.begin(), mat._order_of_dimension.end() },
            _size_of_subdimension{ mat._size_of_subdimension.begin(), mat._size_of_subdimension.end() },
            _data{ mat.begin(), mat.end() }
        {

        }

        constexpr const auto operator[] (const size_t index) const noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
        {
            return const_subdimension<T, Rank - 1>
                (
                    _order_of_dimension.subspan<1, Rank - 1>(),
                    _size_of_subdimension.subspan<1, Rank - 1>(),
                    _data.subspan(index * _size_of_subdimension[1], _size_of_subdimension[1])
                );
        }

        constexpr const T& operator[] (const size_t index) const noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
        {
            return _data[index];
        }

        constexpr const auto begin() const noexcept
        {
            return _data.begin();
        }

        constexpr const auto end() const noexcept
        {
            return _data.end();
        }

        constexpr const auto cbegin() const noexcept
        {
            return _data.begin();
        }

        constexpr const auto cend() const noexcept
        {
            return _data.end();
        }

        constexpr auto get_Rank() const noexcept
        {
            return _order_of_dimension;
        }

        constexpr const size_t& order_of_dimension(const size_t& index) const noexcept
        {
            return _order_of_dimension[index];
        }

        constexpr const size_t& size_of_subdimension(const size_t& index) const noexcept
        {
            return _size_of_subdimension[index];
        }

        constexpr const size_t& order_of_current_dimension() const noexcept
        {
            return _order_of_dimension[0];
        }

        constexpr const size_t& size_of_current_tensor() const noexcept
        {
            return _size_of_subdimension[0];
        }

        constexpr const T* data() const noexcept
        {
            auto it = _data.begin();
            return reinterpret_cast<const T*>(&it);
        }
    };

    template <typename T, size_t Rank>
    class subdimension
    {
        using SourceSizeOfDimensionArraySpan        = std::span<size_t, Rank>;
        using SourceSizeOfSubdimensionArraySpan     = std::span<size_t, Rank>;
        using SourceDataSpan                        = std::span<T>;
        
    private:
        SourceSizeOfDimensionArraySpan              _order_of_dimension;
        SourceSizeOfSubdimensionArraySpan           _size_of_subdimension;
        SourceDataSpan                              _data;

    public:
        friend class const_subdimension<T, Rank>;
        friend class tensor<T, Rank>;

        constexpr subdimension() = delete;
        constexpr subdimension(subdimension&&) noexcept = delete;
        constexpr subdimension(const subdimension&) noexcept = default;
        constexpr subdimension(const const_subdimension<T, Rank>&) noexcept = delete;

        constexpr subdimension(const SourceSizeOfDimensionArraySpan& size_of_dimension_span, const SourceSizeOfSubdimensionArraySpan& size_of_subdimension_span, const SourceDataSpan& data_span) noexcept :
            _order_of_dimension{ size_of_dimension_span },
            _size_of_subdimension{ size_of_subdimension_span },
            _data{ data_span }
        {

        }

        constexpr subdimension(tensor<T, Rank>& mat) noexcept :
            _order_of_dimension{ mat._order_of_dimension.begin(), mat._order_of_dimension.end() },
            _size_of_subdimension{ mat._size_of_subdimension.begin(), mat._size_of_subdimension.end() },
            _data { mat.begin(), mat.end() }
        {

        }

        constexpr auto& operator=(const useful_specializations::nested_initializer_list<T, Rank>& data) requires useful_concepts::is_greater_than<size_t, size_t, Rank, 2>
        {
            if (order_of_current_dimension() != data.size())
                throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

            auto it = data.begin();

            for (size_t index = 0; index < order_of_current_dimension(); index++)
            {
                (*this)[index] = *(it + index);
            }

            return (*this);
        }

        constexpr auto& operator=(const std::initializer_list<std::initializer_list<T>>& data) requires useful_concepts::is_equal_to<size_t, size_t, Rank, 2>
        {
            if (order_of_current_dimension() != data.size())
                throw std::runtime_error("Size of initializer_list doesn't match size of dimension!");

            auto it = data.begin();

            for (size_t index = 0; index < order_of_current_dimension(); index++)
            {
                (*this)[index] = *(it + index);
            }

            return (*this);
        }

        constexpr auto& operator=(const std::initializer_list<T>& data) 
        {
            if (size_of_current_tensor() != data.size())
                throw std::runtime_error("Size of initializer_list doesn't match size of tensor!");

            std::copy(data.begin(), data.end(), begin());

            return (*this);
        }

        constexpr auto operator[] (const size_t index) noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
        {
            return subdimension<T, Rank - 1>
                (
                    
                    _order_of_dimension.subspan<1, Rank - 1>(),
                    _size_of_subdimension.subspan<1, Rank - 1>(),
                    _data.subspan(index * _size_of_subdimension[1], _size_of_subdimension[1])
                );
        }

        constexpr const auto operator[] (const size_t index) const noexcept requires useful_concepts::is_greater_than<size_t, size_t, Rank, 1>
        {
            return const_subdimension<T, Rank - 1>
                (
                    std::span<const size_t, Rank - 1> (_order_of_dimension.begin() + 1, _order_of_dimension.begin() + Rank),
                    std::span<const size_t, Rank - 1> (_size_of_subdimension.begin() + 1, _size_of_subdimension.begin() + Rank),
                    std::span<const T>(begin() + index * _size_of_subdimension[1], begin() + index * _size_of_subdimension[1] + _size_of_subdimension[1])
                );
        }

        constexpr const T& operator[] (const size_t index) const noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
        {
            return _data[index];
        }

        constexpr T& operator[] (const size_t index) noexcept requires useful_concepts::is_equal_to<size_t, size_t, Rank, 1>
        {
            return _data[index];
        }

        constexpr auto begin() noexcept
        {
            return _data.begin();
        }

        constexpr const auto begin() const noexcept
        {
            return std::span<const T>(_data).begin();
        }

        constexpr const auto cbegin() const noexcept
        {
            return std::span<const T>(_data).begin();
        }

        constexpr auto end() noexcept
        {
            return _data.end();
        }

        constexpr const auto end() const noexcept
        {
            return std::span<const T>(_data).end();
        }

        constexpr const auto cend() const noexcept
        {
            return std::span<const T>(_data).end();
        }

        constexpr auto get_Rank() const noexcept
        {
            std::span<const size_t> dims_sizes{ _order_of_dimension.begin(), _order_of_dimension.end() };

            return dims_sizes;
        }

        constexpr const size_t& order_of_dimension(const size_t& index) const noexcept
        {
            return _order_of_dimension[index];
        }

        constexpr const size_t& size_of_subdimension(const size_t& index) const noexcept
        {
            return _size_of_subdimension[index];
        }

        constexpr const size_t& order_of_current_dimension() const noexcept
        {
            return _order_of_dimension[0];
        }

        constexpr const size_t& size_of_current_tensor() const noexcept
        {
            return _size_of_subdimension[0];
        }

        constexpr T* data() noexcept
        {
            auto it = _data.begin();
            return reinterpret_cast<T*>(&it);
        }
    };

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