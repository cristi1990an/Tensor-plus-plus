# Tensor-plus-plus

## Functionality
	
**tensor_lib::tensor** is an allocator-aware template container that describes a mathematical tensor, implemented as a heap allocated array. It is build as an alternative to structures such as std::vector<std::vector<std::vector<...>>>, replicating its behaviour and syntax almost entirely. The improvements (and the implementational challenges) come from the fact that our tensor's underlyting data is contiguous in memory as oppossed to the "all over the place" allocated data in a nested vector structure. 

**Rational:** The decision of having the whole data allocated contiguously is a "no-brainer", allowing for fast and easy allocation, accessing, copying, moving etc. Open-source alternatives to nested vector already exist, both in the form of 2d matrices and multi-dimensional tensors such as ours. One major caveat they all share and reason why certain developers are still inclined to keep to nested vector structures, even in modern neural-network projects, has always been the "syntactic sugar" such structures provide.
	
A syntax feature that a nested std::vector structure has and that was necessary to emulate from the get-go was the way it could interpret/initialize/assign from a nested initializer_list structure, having each layer/rank in the structure have a constructor taking an std::initializer_list<T>, creating a perfect match between:
	
```
std::vector		<std::vector		<std::vector		<T>>>	and...
std::initializer_list	<std::initializer_list	<std::initializer_list	<T>>>
```
	
```
std::vector <std::vector <std::vector<int>>> nested_vec =
{
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3},
  },
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3}
  },
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3},
  },
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3}
  }
};
```


Particularly useful being the copy assignment operator working on a per-dimension basis...

```
  nested_vec[1] = { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } };
```

tensor_lib::tensor offers the same functionality while being faster to initialize.

```
tensor<int, 3> three_dim_tensor =
{
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3},
  },
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3}
  },
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3},
  },
  {
    {1, 2, 3}, {1, 2, 3}, {1, 2, 3}
  }
};

three_dim_tensor[1] = { { 1, 2, 3 }, { 1, 2, 3 }, { 1, 2, 3 } };
```
  
Another feature tensor implements is being able to stack calls of the '[]' operator relative to each dimension.
Here we have a tensor with 3 dimension. Each of these dimensions has 4 subdimensions. Each of these subdimension has 5 sub-subdimensions and so on...
	
Let's start by explicitly creating a 5-dimensional tensor of sizes 3 by 4 by 5 by 6 by 7:

```
tensor<int, 5> my_tensor( 3u, 4u, 5u, 6u, 7u );
```
And then iterate through the whole tensor using nested for-loops: 
```
int val = 0;

for (size_t a = 0; a < my_tensor.order_of_dimension(0) /* returns 3 */; a++)
{
  for (size_t b = 0; b < my_tensor.order_of_dimension(1) /* returns 4 */; b++)
  {
    for (size_t c = 0; c < my_tensor.order_of_dimension(2) /* returns 5 */; c++)
    {
      for (size_t d = 0; d < my_tensor.order_of_dimension(3) /* returns 6 */; d++)
      {
        for (size_t e = 0; e < my_tensor.order_of_dimension(4) /* returns 7 */; e++)

          my_tensor[a][b][c][d][e] = val++;  // We can do this <3
      }
    }
  }
}
```
	
## Implementation
	
	
## Template parameters
	
**tensor_lib::tensor<T, Rank, typename allocator_type = = std::allocator<std::remove_cv_t<T>>>**

**T**			-	type of the elements

**Rank**		-	the rank of the tensor, aka the number of dimensions

**allocator_type	- 	an allocator that is used to acquire/release memory and to construct/destroy the elements in that memory; defaults to the default allocator of T

One specific feature of the tensor class is the ability of having intuitive syntax when stacking calls to the operator[] and being able to interpret nested initializer_list structures like in the examples above.
The way it works is that operator[] returns an instance of "subdimension<Rank - 1>", a lightweight instance of an object that referes to the data owned by the parent tensor. It's implemented using a dynamic span of the original data it covers, a static span of the sizes it needs and a static span of the array with precomputed sizes of the submatrices at each dimension, which in turn returns the same and so on. sizeof(subdimension) being always the size of 4 pointers.

```
std::cout << "Size of the subdimension instance: " << sizeof(my_tensor[0]) << '\n';
std::cout << "Size of 4 pointers: " << 4 * sizeof(void*) << "\n \n";
```


There are no copies involved, though random access calls using this method (when not using an iterator on the whole data) is slower than
the pointer dereferencing done by a nested vector structure because operator[] calculates the resulting range in the _data buffer and constructs an instance of subdimension. On the bright side, we're providing standard iteration capabilities, not only through the whole tensor, but through each subdimension at any of the ranks.
A tensor's memory is contiguous in memory and we can take advantage of far greater performance when iterating through our data this way. Preferably, users can use the square paranthesis operator to calculate the value range representing their desired subdimension and then access its value like a normal array.
In the example below we're iterating through the whole first subdimension of the second rank of the tensor, setting each value to zero.

```
for (auto& val : my_tensor[0][0]) 
{
  val = 0;
}
```

	
Tensor is also compatible with standard algorithms. 
	
```
std::fill(my_tensor[2][1].begin(), my_tensor[2][1].end(), 0);
std::sort(my_tensor[2][1].begin(), my_tensor[2][1].end());
```

	
The design of the complementary non-owning subdimension class template requires however the implementation of a const_subdimension class template in order to maintain const correctness. This is similar to the behaviour of std::iterator and std::const_iterator. const_subdimension being forbidden from changing the data of the parent tensor object, while a const subdimension only restricting its own reassignment to a different range (which is normally possible).

const_subdimension will be returned by the operator[] of either 'tensor' or 'subdimension' when called in a const context.
	
```
[]<typename T, size_t Rank>(const tensor<T, Rank>& my_tensor)
{
  auto subdim = my_tensor[0];
  std::cout << typeid(subdim).name() << '\n\n'; // const_subdimension

  // subdim[0][0][0][0] = 5;' won't work since const_subdimension can't modify the underlying data. It is however allowed to assign itself to other spans, the same way
  // a const_iterator is allowed to increment or reasign itself to another address.

}(my_tensor);
	
```
	
## Member functions
	
### Constructors
	
### tensor() noexcept;
	
This is the default constructor. It will consider that all subdimensions to be of size 1, effectively making it a one element tensor. Current design considers a tensor that 
has even one subdimension be of size zero to be in an invalid state, so even when default initialized, a tensor will allocate memory for one element (this might change in
the future though...).
	
### template<typename... Sizes> tensor (const Sizes ... sizes) noexcept;

Constructor taking the sizes of each subdimension. Passing "2, 3, 4" into this constructor for example will create a tensor of size 2 by 3 by 4, with a total size of 24 elements, all default initialized. 

The constructor has static requirements that: 
* the number of parameters passed equals the rank of the tensor
* each element in the parameter pack is an integral type
	
Example: 
	
```
tensor<int, 3> tsor (2, 3, 4);
```
	
### tensor(const tensor& other) noexcept;
	
Copy constructor. The tensor will take all the properties of the other tensor (size/dimensions) and make a copy of all elements.

### tensor(const subdimension<T, Rank>& subdimension) noexcept;
	
Copy constructor from a subdimension of the same rank.
	
### tensor(tensor&& other) noexcept;
	
Move constructor. Tensor will take ownership over the heap-allocated data and will reset the other tensor to the default state described by the default constructor.
	
### tensor(const std::initializer_list<T>& data) TENSORLIB_NOEXCEPT_IN_RELEASE;
	
Constructor exclusive to tensor<T, 1u> (one dimensional tensor), taking an initializer_list<T>. The size of the tensor will be equal to the size of the initializer_list. Example:
	
```
tensor<int, 1> tsor = { 1, 2, 3, 4, 5, 6 };
```
	
### tensor(const useful_specializations::nested_initializer_list<T, Rank>& data) TENSORLIB_NOEXCEPT_IN_RELEASE;

Constructor taking a nested initializer_list construct (initializer_list<initializer_list<...initializer_list<T>...>>).

The constructor has static requirements that: 
* the depth of the nested initializer_list (the number of dimensions) must be equal to the rank of the tensor
* the underlying element type must be T

The constructor has run-time requirements (enabled by default with the TENSORLIB_DEBUGGING flag) that:
* subdimensions of the same rank must also be of the same size, aka this is invalid: `{ {1, 2, 3}, {5, 6} }`

### template<typename ... Args>
### tensor(Args&& ... args) TENSORLIB_NOEXCEPT_IN_RELEASE
	
Constructor that supports emplace initialization of all elements in the tensor from a set of perfect forwarded arguments. Example:

```
tensor<std::string, 3> tsor (2, 3, 4, "Cristi");
```
The first N parameters (where N equals the rank of the tensor) are considered the sizes of the tensor and the same constraints are applied to them as with the normal constructor that only takes sizes. All other arguments passed are going to be perfect forwarded to a temporary object with which and the copy constructor of each element in the tensor is going to be called.
	
The constructor has static requirements that: 
* the number of parameters passed must be greater than the rank of the tensor (sizes + arguments)
* the first N (N = Rank) elements in the parameter pack are all of an integral type
* T can be constructed from the arguments passed after the sizes
