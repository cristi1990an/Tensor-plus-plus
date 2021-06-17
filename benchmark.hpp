#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include "tensor.hpp"

namespace benchmark
{
	using namespace tensor_lib;
	#define nested_initializer_list {{{{ 10, 11 },{ 12, 13 },{ 14, 15 },},{ { 16, 17 }, { 18, 19 }, { 20, 21 }, }, { { 22, 23 }, { 24, 25 }, { 26, 27 }, }, { { 28, 29 }, { 30, 31 }, { 32, 33 }, } }, { { { 34, 35 }, { 36, 37 }, { 38, 39 }, }, { { 40, 41 }, { 42, 43 }, { 44, 45 }, }, { { 46, 47 }, { 48, 49 }, { 50, 51 }, }, { { 52, 53 }, { 54, 55 }, { 56, 57 }, } }, { { { 58, 59 }, { 60, 61 }, { 62, 63 }, }, { { 64, 65 }, { 66, 67 }, { 68, 69 }, }, { { 70, 71 }, { 72, 73 }, { 74, 75 }, }, { { 76, 77 }, { 78, 79 }, { 80, 81 }, } }, { { { 82, 83 }, { 84, 85 }, { 86, 87 }, }, { { 88, 89 }, { 90, 91 }, { 92, 93 }, }, { { 94, 95 }, { 96, 97 }, { 98, 99 }, }, { { 10, 11 }, { 12, 13 }, { 14, 15 }, } }, { { { 16, 17 }, { 18, 19 }, { 20, 21 }, }, { { 22, 23 }, { 24, 25 }, { 26, 27 }, }, { { 28, 29 }, { 30, 31 }, { 32, 33 }, }, { { 34, 35 }, { 36, 37 }, { 38, 39 }, } } }
	#define empty_nested_initializer_list {{{{ 0, 0 },{ 0, 0 },{ 0, 0 },},{ { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } }, { { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, }, { { 0, 0 }, { 0, 0 }, { 0, 0 }, } } }


	#define ITERATIONS 200000

	void BENCHMARK_ALLOCATION_FROM_NESTED_INITIALIZER_LIST()
	{
		unsigned long long average_time = 0;
		
		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<int, 4> tsor( 5u, 4u, 3u, 2u );
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

			std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;

			stop = std::chrono::high_resolution_clock::now();

			average_time += (stop - start).count();
		}

		average_time /= ITERATIONS;

		std::cout << "\tVector average initialization time: " << average_time << "\n";
		std::cout << '\n';
	}

	void BENCHMARK_BIG_ALLOCATION()
	{
		unsigned long long average_time = 0;

		for (unsigned int i = 0; i < ITERATIONS; i++)
		{
			std::chrono::high_resolution_clock::time_point start, stop;

			start = std::chrono::high_resolution_clock::now();

			tensor<int, 1> tsor( 100000u );

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
		unsigned long long tensor_average_time = 0;
		unsigned long long vector_average_time = 0;
		tensor<int, 4> tsor{ 5, 4, 3, 2 };
		tsor = nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;
		size_t i, j, k, l;
		int val;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			i = std::rand() % 5;
			j = std::rand() % 4;
			k = std::rand() % 3;
			l = std::rand() % 2;
			val = std::rand() % std::_Max_possible_v<int>;

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
		unsigned long long tensor_average_time = 0;
		unsigned long long vector_average_time = 0;
		tensor<int, 4> tsor{ 5, 4, 3, 2 };
		tsor = nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;
		size_t i, j, k, l, arr_it;
		std::array<int, 5 * 4 * 3 * 2> data;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			std::generate(data.begin(), data.end(), []() {return std::rand() % std::_Max_possible_v<int>; });
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
		unsigned long long tensor_average_time = 0;
		unsigned long long vector_average_time = 0;
		tensor<int, 4> tsor{ 5, 4, 3, 2 };
		tsor = nested_initializer_list;
		std::vector<std::vector<std::vector<std::vector<int>>>> vec nested_initializer_list;
		std::chrono::high_resolution_clock::time_point start, stop;
		std::array<int, 5 * 4 * 3 * 2> data;
		size_t i, j, k, l, arr_it;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			std::generate(data.begin(), data.end(), []() {return std::rand() % std::_Max_possible_v<int>; });
			
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
		unsigned long long tensor_average_time = 0;
		unsigned long long vector_average_time = 0;
		tensor<int, 1> tsor{ 1000 };
		std::vector<int> vec;
		vec.reserve(1000);
		std::chrono::high_resolution_clock::time_point start, stop;
		size_t i;
		std::array<int, 1000> data;


		for (unsigned int it = 0; it < ITERATIONS; it++)
		{
			std::generate(data.begin(), data.end(), []() {return std::rand() % std::_Max_possible_v<int>; });
			start = std::chrono::high_resolution_clock::now();
			for (i = 0; i < 1000; i++)
				tsor[i] = data[i++];
			stop = std::chrono::high_resolution_clock::now();

			tensor_average_time += (stop - start).count();

			start = std::chrono::high_resolution_clock::now();
			for (i = 0; i < 1000; i++)
				vec[i] = data[i++];
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
		unsigned long long tensor_average_time = 0;
		unsigned long long vector_average_time = 0;
		tensor<int, 4> tsor{ 5, 4, 3, 2 }, destination_tsor{ 5, 4, 3, 2 };
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

	void BENCHMARK_AGAINST_VECTOR()
	{
		std::cout << "\nBenchmarking against vector...\n\n";

		BENCHMARK_ALLOCATION_FROM_NESTED_INITIALIZER_LIST();
		BENCHMARK_BIG_ALLOCATION();
		BENCHMARK_RANDOM_ACCESS();
		BENCHMARK_ASSIGN_THROUGH_BRACKETS();
		BENCHMARK_ASSIGN_THROUGH_ITERATOR();
		BENCHMARK_ASSIGN_ONE_DIMENSION();
		BENCHMARK_COPY();

		std::cout << '\n';
	}

	void RUN_ALL()
	{
		BENCHMARK_AGAINST_VECTOR();
	}
}
