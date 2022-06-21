#include "../inc/tensor.hpp"

#include <algorithm>

using namespace tensor_lib;

void print_tensor(tensor_object auto&& tsor)
{
	static int C = 3;

	std::cout << '[' << C++ << "]\n\n";

	for (std::size_t i = 0; i < tsor.order_of_dimension(0); i++)
	{
		for (std::size_t j = 0; j < tsor.order_of_dimension(1); j++)
		{
			std::cout << "{ ";

			for (std::size_t k = 0; k < tsor.order_of_dimension(2); k++)
			{
				std::cout << tsor[i][j][k] << ' ';
			}

			std::cout << "}, ";
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

int main()
{
	tensor<int, 3> tsor =
	{
		{
			{ 10, 11, 12, 13, 14 }, { 15, 16, 17, 18, 19 }, { 20, 21, 22, 23, 24 }
		},
		{
			{ 10, 11, 12, 13, 14 }, { 15, 16, 17, 18, 19 }, { 20, 21, 22, 23, 24 }

		},
		{
			{ 10, 11, 12, 13, 14 }, { 15, 16, 17, 18, 19 }, { 20, 21, 22, 23, 24 }
		},
		{
			{ 10, 11, 12, 13, 14 }, { 15, 16, 17, 18, 19 }, { 20, 21, 22, 23, 24 }
		}
	};

	/* [1] Dublam toate valorile din a doua subdimensiune de ordinul 2 */

	auto subdimensiune = tsor[1];

	for (auto& valoare : subdimensiune)
	{
		valoare = valoare * 2;
	}

	/* [2] Din acea subdimensiune, afisam a doua subdimensiune de ordinul 1 */

	std::cout << "[2]\n\n{ ";
	for (const auto valoare : subdimensiune[1])
	{
		std::cout << valoare << ' ';
	}
	std::cout << "}\n\n";

	/* [3] Umplem ultimele doua subdimensiuni de ordinul 2 din tensor cu 42 */

	std::fill(tsor[2].begin(), tsor[3].end(), 42);

	print_tensor(tsor);

	/* [4] Interschimbam prima si ultima subdimensiune de ordinul 2 */

	swap(tsor[0], tsor[3]);

	print_tensor(tsor);

	/* [5] Asignam valori noi celei de a treia subdimensiuni */

	tsor[2] = {
		{ 77, 77, 77, 77, 77 }, { 88, 88, 88, 88, 88 }, { 99, 99, 99, 99, 99 }
	};

	print_tensor(tsor);
}