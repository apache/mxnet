#ifndef _LOG_UNIFORM_SAMPLER_H
#define _LOG_UNIFORM_SAMPLER_H

#include <unordered_set>
#include <vector>
#include <utility>
#include <random>

class Log_Uniform_Sampler
{
	private:
		const int N;
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution;
		std::vector<float> prob;

	public:
		Log_Uniform_Sampler(const int);
		float probability(const int);
		std::vector<float> expected_count(const int, std::vector<long>);
        std::vector<std::pair<long, long>> accidental_matches(std::vector<long>, std::vector<long>);
		std::unordered_set<long> sample(const size_t, int*);
		std::unordered_set<long> sample_unique(const size_t, std::unordered_set<long>);
};

#endif // _LOG_UNIFORM_SAMPLER_H
