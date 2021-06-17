# Tensor-plus-plus

```
template<typename T, std::size_t Size>
class Template;
```

tensor_lib::tensor is a class template that describes a mathematical tensor, implemented as a heap allocated array. It is build as an alternative
to structures such as std::vector<std::vector<std::vector<...>>>, replicating its behaviour and syntax almost entirely with the added benefits of having its data
contiguous in memory. 

Example: Here we have a tensor with 3 dimension. Each of these dimensions has 4 subdimensions. Each of these subdimension has 5 sub-subdimensions and so on...

```
tensor<int, 5> my_tensor{ 3, 4, 5, 6, 7 }; // 5-dimensional tensor of size (3 * 4 * 5 * 6 * 7)

int val = 0;

for (size_t a = 0; a < my_tensor.order_of_dimension(0) /* return 3 */; a++)
{
  for (size_t b = 0; b < my_tensor.order_of_dimension(1) /* return 4 */; b++)
  {
    for (size_t c = 0; c < my_tensor.order_of_dimension(2) /* return 5 */; c++)
    {
      for (size_t d = 0; d < my_tensor.order_of_dimension(3) /* return 6 */; d++)
      {
        for (size_t e = 0; e < my_tensor.order_of_dimension(4) /* return 7 */; e++)

          my_tensor[a][b][c][d][e] = val++;  // We can do this <3
      }
    }
  }
}
```

Template parameters

T		-	type of the elements
Rank	-	the rank of the tensor, aka the number of dimensions

One specific feature of the tensor class is the ability of having intuitive syntax when stacking calls to the operator[] like in the example above.
The way it works is that operator[] returns an instance of "subdimension<Rank - 1>", a lightweight instance of an object that referes to the data owned by the parent 
tensor. It's implemented using a dynamic span of the original data it covers, a static span of the sizes it needs and a static span of the array with precomputed sizes of 
the submatrices at each dimension, which in turn returns the same and so on. sizeof(subdimension) being always the size of 4 pointers.

```
std::cout << "Size of the subdimension instance: " << sizeof(my_tensor[0]) << '\n';
std::cout << "Size of 4 pointers: " << 4 * sizeof(void*) << "\n \n";
```

There are no copies involved, though random access calls using this method (when not using an iterator on the whole data) is slower than
the pointer dereferencing done by a nested vector structure because operator[] calculates the resulting range in the _data buffer and constructs an instance of 
subdimension. We're however making up for it by providing standard iteration capabilities, not only through the whole tensor,
but through each subdimension at any of the ranks. In the example below we're iterating through the whole first subdimension of the first rank of the tensor, setting 
each value to zero.

```
for (auto& val : my_tensor[0]) 
{
  val = 0;
}
```

The design of the complementary non-owning subdimension class template requires however the implementation of a const_subdimension class template in order to maintain
const correctness. This is similar to the behaviour of std::iterator and std::const_iterator. const_subdimension being forbidden from changing the data of the parent
tensor object, while a const subdimension only restricting its own reassignment to a different range (which is normally possible).


const_subdimension will be returned by the operator[] of either 'tensor' or 'subdimension' when called in a const context.

```
[]<typename T, size_t Rank>(const tensor<T, Rank>& my_tensor)
{
  auto subdim = my_tensor[0];
  std::cout << typeid(subdim).name() << '\n\n';

  // subdim[0][0][0][0] = 5;' won't work 

}(my_tensor);
```
