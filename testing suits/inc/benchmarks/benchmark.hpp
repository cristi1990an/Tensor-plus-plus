#pragma once

#include "../../../inc/tensor.hpp"

#include <chrono>
#include <iostream>
#include <vector>

namespace benchmark
{
	using namespace tensor_lib;
#define nested_initializer_list {{{{ 10, 11 },{ 12, 13 },{ 14, 15 },},{ { 16, 17 }, { 18, 19 }, { 20, 21 }, }, { { 22, 23 }, { 24, 25 }, { 26, 27 }, }, { { 28, 29 }, { 30, 31 }, { 32, 33 }, } }, { { { 34, 35 }, { 36, 37 }, { 38, 39 }, }, { { 40, 41 }, { 42, 43 }, { 44, 45 }, }, { { 46, 47 }, { 48, 49 }, { 50, 51 }, }, { { 52, 53 }, { 54, 55 }, { 56, 57 }, } }, { { { 58, 59 }, { 60, 61 }, { 62, 63 }, }, { { 64, 65 }, { 66, 67 }, { 68, 69 }, }, { { 70, 71 }, { 72, 73 }, { 74, 75 }, }, { { 76, 77 }, { 78, 79 }, { 80, 81 }, } }, { { { 82, 83 }, { 84, 85 }, { 86, 87 }, }, { { 88, 89 }, { 90, 91 }, { 92, 93 }, }, { { 94, 95 }, { 96, 97 }, { 98, 99 }, }, { { 10, 11 }, { 12, 13 }, { 14, 15 }, } }, { { { 16, 17 }, { 18, 19 }, { 20, 21 }, }, { { 22, 23 }, { 24, 25 }, { 26, 27 }, }, { { 28, 29 }, { 30, 31 }, { 32, 33 }, }, { { 34, 35 }, { 36, 37 }, { 38, 39 }, } } }
#define empty_nested_initializer_list {{{{ 0, 0 },{ 0, 0 },{ 0, 0 },},{ { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } } }
#define one_dimensional_initializer_list { 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9 }
#define non_trivial_one_dimensional_initializer_list {"Some long string can can't be optimized ... 1","Some long string can can't be optimized ... 2","Some long string can can't be optimized ... 3","Some long string can can't be optimized ... 4","Some long string can can't be optimized ... 5","Some long string can can't be optimized ... 6","Some long string can can't be optimized ... 7","Some long string can can't be optimized ... 8","Some long string can can't be optimized ... 9","Some long string can can't be optimized ... 10","Some long string can can't be optimized ... 11","Some long string can can't be optimized ... 12"}

#define ITERATIONS 200000

	void BENCHMARK_ALLOCATION_FROM_NESTED_INITIALIZER_LIST_WITH_EXPLICIT_SIZES()
	{
		long long average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<int, 4> tsor(5, 4, 3, 2);
			tsor = nested_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tTensor average initialization time: " << average_time << "\n";

		average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			std::vector<std::vector<std::vector<std::vector<int>>>> vec = nested_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tVector average initialization time: " << average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_ALLOCATION_FROM_NESTED_INITIALIZER_LIST_WITH_DEDUCED_SIZES()
	{
		long long average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<int, 4> tsor = nested_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tTensor average initialization time when deducing sizes: " << average_time << "\n";

		average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			std::vector<std::vector<std::vector<std::vector<int>>>> vec = nested_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tVector average initialization time (always deducing sizes): " << average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_ALLOCATION_FROM_ONE_DIMENSIONAL_INITIALIZER_LIST()
	{
		long long average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<int, 1> tsor = one_dimensional_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tTensor average initialization time from one dimensional initializer list: " << average_time << "\n";

		average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			std::vector<int> vec = one_dimensional_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tVector average initialization time from one dimensional initializer list: " << average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_ALLOCATION_FROM_NON_TRIVIAL_ONE_DIMENSIONAL_INITIALIZER_LIST()
	{
		long long average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<std::string, 1> tsor = non_trivial_one_dimensional_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tTensor average initialization time from non-trivial one dimensional initializer list: " << average_time << "\n";

		average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			std::vector<std::string> vec = non_trivial_one_dimensional_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tVector average initialization time from non-trivial one dimensional initializer list: " << average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_BIG_ALLOCATION()
	{
		long long average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<int, 1> tsor(100000);

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tTensor average allocation time: " << average_time << "\n";

		average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			std::vector<int> vec;
			vec.reserve(100000);

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tVector average allocation time: " << average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_RANDOM_ACCESS()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		tensor<int, 4> tsor(5, 4, 3, 2);
		tsor = nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;
		size_t i, j, k, l;
		int val;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			i = static_cast<std::size_t>(std::rand() % 5);
			j = static_cast<std::size_t>(std::rand() % 4);
			k = static_cast<std::size_t>(std::rand() % 3);
			l = static_cast<std::size_t>(std::rand() % 2);
			val = std::rand() % std::numeric_limits<int>::max();

			start = std::chrono::high_resolution_clock::now();

			tsor[i][j][k][l] = val;
			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			start = std::chrono::high_resolution_clock::now();
			vec[i][j][k][l] = val;
			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();
		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tTensor average random access time: " << tensor_average_time << "\n";
		std::cout << "\tVector average random access time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_ASSIGN_THROUGH_BRACKETS()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		tensor<int, 4> tsor(5, 4, 3, 2);
		tsor = nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;
		size_t i, j, k, l, arr_it;
		std::array<int, 5 * 4 * 3 * 2> data;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			std::generate(data.begin(), data.end(), []() {return std::rand() % std::numeric_limits<int>::max(); });
			arr_it = 0;

			start = std::chrono::high_resolution_clock::now();

			for (i = 0; i < 5; i++)
				for (j = 0; j < 4; j++)
					for (k = 0; k < 3; k++)
						for (l = 0; l < 2; l++)
							tsor[i][j][k][l] = data[arr_it];

			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();
			arr_it = 0;

			start = std::chrono::high_resolution_clock::now();

			for (i = 0; i < 5; i++)
				for (j = 0; j < 4; j++)
					for (k = 0; k < 3; k++)
						for (l = 0; l < 2; l++)
							vec[i][j][k][l] = data[arr_it];

			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();

			arr_it++;

		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tTensor average fill through brackets time: " << tensor_average_time << "\n";
		std::cout << "\tVector average fill through brackets time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_ASSIGN_THROUGH_ITERATOR()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		tensor<int, 4> tsor(5, 4, 3, 2);
		tsor = nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;
		std::array<int, 5 * 4 * 3 * 2> data;
		size_t i, j, k, l, arr_it;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			std::generate(data.begin(), data.end(), []() {return std::rand() % std::numeric_limits<int>::max(); });

			start = std::chrono::high_resolution_clock::now();
			std::copy(data.cbegin(), data.cend(), tsor.begin());
			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			arr_it = 0;

			start = std::chrono::high_resolution_clock::now();
			for (i = 0; i < 5; i++)
				for (j = 0; j < 4; j++)
					for (k = 0; k < 3; k++)
						for (l = 0; l < 2; l++)
						{
							vec[i][j][k][l] = data[arr_it++];
						}
			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();
		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tTensor average fill through iterator time: " << tensor_average_time << "\n";
		std::cout << "\tVector average fill through iterator time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_ASSIGN_ONE_DIMENSION()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		tensor<int, 1> tsor(1000);
		std::vector<int> vec(1000);
		std::chrono::high_resolution_clock::time_point start, stop;
		size_t i;
		std::array<int, 1000> data;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			std::generate(data.begin(), data.end(), []() {return std::rand() % std::numeric_limits<int>::max(); });
			start = std::chrono::high_resolution_clock::now();
			for (i = 0; i < 1000; i++)
				tsor[i] = data[i];
			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			start = std::chrono::high_resolution_clock::now();
			for (i = 0; i < 999; i++)
				vec[i] = data[i];
			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();
		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tOne dimensional tensor average fill through brackets time: " << tensor_average_time << "\n";
		std::cout << "\tOne dimensional vector average fill through brackets time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_COPY()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		tensor<int, 4> tsor(5, 4, 3, 2), destination_tsor(5, 4, 3, 2);
		tsor = nested_initializer_list;
		destination_tsor = empty_nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> destination_vec empty_nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;

		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			start = std::chrono::high_resolution_clock::now();
			std::copy(tsor.cbegin(), tsor.cend(), destination_tsor.begin());
			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			start = std::chrono::high_resolution_clock::now();
			std::copy(vec.cbegin(), vec.cend(), destination_vec.begin());
			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();
		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tTensor average copy time: " << tensor_average_time << "\n";
		std::cout << "\tVector average copy time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_EMPLACE_INITIALIZATION()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		std::chrono::high_resolution_clock::time_point start, stop;

		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			start = std::chrono::high_resolution_clock::now();

			tensor<std::string, 4> tsor(5, 4, 3, 2, "Some long string I'm writing out of the top of my head...");

			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			start = std::chrono::high_resolution_clock::now();
			std::vector<std::string> vec;
			vec.reserve(5 * 4 * 3 * 2);
			for (size_t index = 0; index != vec.capacity(); index++)
			{
				vec.emplace_back("Some long string I'm writing out of the top of my head...");
			}
			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();
		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tTensor average emplace initialization time: " << tensor_average_time << "\n";
		std::cout << "\tVector average emplace initialization time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_RESIZE()
	{
		long long tensor_average_time = 0;
		long long vector_average_time = 0;
		tensor<int, 4> tsor(1, 1, 1, 1); // also default state
		std::vector<int> vec;
		vec.reserve(1);
		std::chrono::high_resolution_clock::time_point start, stop;

		volatile std::size_t a = 5, b = 7, c = 9, d = 8;

		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			start = std::chrono::high_resolution_clock::now();
			tsor.resize(a, b, c, d);
			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			start = std::chrono::high_resolution_clock::now();
			vec.reserve(a * b * c * d);
			stop = std::chrono::high_resolution_clock::now();

			vector_average_time += (stop - start).count();

			vec.resize(1);
			vec.shrink_to_fit();
			tsor.resize(1, 1, 1, 1);
		}

		tensor_average_time /= ITERATIONS;
		vector_average_time /= ITERATIONS;

		std::cout << "\tTensor average resize time: " << tensor_average_time << "\n";
		std::cout << "\tVector average resize time: " << vector_average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_AGAINST_VECTOR()
	{
		std::cout << "\nBenchmarking against vector...\n\n";

		BENCHMARK_ALLOCATION_FROM_NESTED_INITIALIZER_LIST_WITH_EXPLICIT_SIZES();
		BENCHMARK_ALLOCATION_FROM_NESTED_INITIALIZER_LIST_WITH_DEDUCED_SIZES();
		BENCHMARK_ALLOCATION_FROM_ONE_DIMENSIONAL_INITIALIZER_LIST();
		BENCHMARK_ALLOCATION_FROM_NON_TRIVIAL_ONE_DIMENSIONAL_INITIALIZER_LIST();
		BENCHMARK_BIG_ALLOCATION();
		BENCHMARK_RANDOM_ACCESS();
		BENCHMARK_ASSIGN_THROUGH_BRACKETS();
		BENCHMARK_ASSIGN_THROUGH_ITERATOR();
		BENCHMARK_ASSIGN_ONE_DIMENSION();
		BENCHMARK_COPY();
		BENCHMARK_RESIZE();
		BENCHMARK_EMPLACE_INITIALIZATION();

		std::cout << '\n';
	}

	void RUN_ALL()
	{
		BENCHMARK_AGAINST_VECTOR();
	}
}
