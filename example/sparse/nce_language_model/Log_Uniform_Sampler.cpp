#include "Log_Uniform_Sampler.h"

#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <stddef.h>
#include <thread>
#include <iostream>

Log_Uniform_Sampler::Log_Uniform_Sampler(const int range_max) : N(range_max), generator(1111), distribution(0.0, 1.0), prob(N, 0)
{
	for(int idx = 0; idx < N; ++idx)
	{
		prob[idx] = (log(idx+2) - log(idx+1)) / log(range_max+1);
	}
}

float Log_Uniform_Sampler::probability(const int idx)
{
	return prob[idx];
}

std::vector<float> Log_Uniform_Sampler::expected_count(const int num_tries, std::vector<long> samples)
{
	std::vector<float> freq;
	for(auto& idx : samples)
	{
		float value = -expm1(num_tries * log1p(-prob[idx]));
		freq.emplace_back(value);
	}
	return freq;
}

std::vector<std::pair<long, long>> Log_Uniform_Sampler::accidental_matches(std::vector<long> labels, std::vector<long> samples)
{
    std::unordered_map<long, long> sample_dict;
    for(size_t index = 0; index < samples.size(); ++index)
	{
        const long& value = samples[index];
		sample_dict[value] = index;
	}

	std::vector<std::pair<long, long>> result;
    for(size_t index = 0; index < labels.size(); ++index)
	{
        const long& value = labels[index];
        auto&& search = sample_dict.find(value);

		if( search != sample_dict.end() )
		{
			result.emplace_back(index, search->second);
		}
	}
	return result;
}

std::unordered_set<long> Log_Uniform_Sampler::sample(const size_t size, int* num_tries)
{
	std::unordered_set<long> data;
	const double log_N = log(N);

	*num_tries = 0;
	while(data.size() != size)
	{
		*num_tries += 1;
		double x = distribution(generator);
		long value = (lround(exp(x * log_N)) - 1);
		data.emplace(value);
	}
	return data;
}

std::unordered_set<long> Log_Uniform_Sampler::sample_unique(const size_t size, std::unordered_set<long> labels)
{
       std::unordered_set<long> data;
       const double log_N = log(N);

       while(data.size() != size)
       {
               double x = distribution(generator);
               long value = (lround(exp(x * log_N)) - 1);
               if( labels.find(value) == labels.end() )
               {
                       data.emplace(value);
               }
       }
       return data;
}
