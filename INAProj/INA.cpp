#include <stdio.h>
#include <functional>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <algorithm>
#include "Barrier.h"
#include <assert.h>
#include <atomic>
#include <string.h>
#include "INAAtomics.h"
#include <math.h>
using namespace std;

#define FLOAT_COUNT 104857600 //400MB
#define ITERATION 10
#define COPIES 1
#define READ_ONLY_SAL
#define READ_WRITE_SAL
#define __OUT__
#define ALIGNED(x) __attribute__((aligned(x)))
#define CACHE_LINE_SIZE 64
#define var auto
#define PROC_CNT thread::hardware_concurrency()
#define MAX_PROC 64

typedef float real_t;
typedef function<void(int, void*)> INAKernel;

READ_ONLY_SAL real_t Inputs[FLOAT_COUNT]  ALIGNED(CACHE_LINE_SIZE); //put garbage in
READ_WRITE_SAL real_t Outputs[FLOAT_COUNT]  ALIGNED(CACHE_LINE_SIZE);

#ifdef __INTELLISENSE__
//peace visual studio
extern bool __sync_bool_compare_and_swap(void* location, long long old, long long desired);
#endif

void GetRange(
	int keyCnt,
	int copyCnt,
	int procCnt,
	int id,
	int& __OUT__ copyId,
	int& __OUT__ start,
	int& __OUT__ end)
{
	//key >> procCnt
	int procPerCopy = procCnt / copyCnt;
	if (procPerCopy > 0)
	{
		//multiple procs work on singe copy
		copyId = id / procPerCopy;
		var chunkId = id % procPerCopy;
		var chunkSize = keyCnt / procPerCopy;
		start = chunkId * chunkSize;
		end = start + chunkSize;
	}
	else
	{
		assert(false);
		exit(-1);
		//1 proc work on multiple copies
	}
}
#pragma region RunNaivePass
typedef union
{
	uint64_t backingStore;
	real_t fv[2];
} TaggedFloat;

TaggedFloat ScratchNaive[FLOAT_COUNT] ALIGNED(CACHE_LINE_SIZE);
#define LOWER_BIT_MASK ((1<<32) - 1)
#define HIGHER_BIT (1 << 32)
void RunNaivePass(int proc, void* additionalArgs)
{
	var start = 0;
	var end = 0;
	var copyId = 0;
	var numProcs = PROC_CNT;
	GetRange(FLOAT_COUNT, COPIES, numProcs, proc, copyId, start, end);
	assert(end != 0);
	for (int i = start; i < end; i++)
	{
		TaggedFloat old, target;
		do
		{
			old = ScratchNaive[i];
			target.fv[1] = old.fv[1] + 1;
			target.fv[0] = old.fv[0] + Inputs[i];
		} 
		while (!__sync_bool_compare_and_swap(&ScratchNaive[i].backingStore, old.backingStore, target.backingStore));
		if (target.fv[1] == numProcs)
		{
			Outputs[i] = target.fv[0];
		}
	}
}
#pragma endregion

#pragma region MyRegion

real_t** ScratchReductionTree ALIGNED(64);
class ReductionTreeArgs
{
public:
	int Base;
};

void RunNaiveReductionTreePass(int proc, void* additionalArgs)
{
	var pArgs = (ReductionTreeArgs*)additionalArgs;
	var base = pArgs->Base;
	var level = (int)(log(proc + 1) / log(base));
	//proc i read 2*i, 2*i+1
	if (level == 0)
	{
		//the finally reduced 

	}
}

#pragma endregion



void TestIterator(
	int iterations,
	vector<INAKernel> kernels,
	vector<void*> args,
	string identifier)
{
	vector<thread> threads(kernels.size());
	Barrier task_barrier(kernels.size());
	for (int i = 0; i < kernels.size(); i++)
	{
		threads[i] = thread(bind(
			[&](int total, int id, void* arg, INAKernel kernel)
		{
			for (int it = 0; it < total; it++)
			{
				kernel(id,arg);
				task_barrier.Wait();
			}
		}, iterations, i, args[i], kernels[i]
			));
	}
	var start = std::chrono::system_clock::now();
	for_each(threads.begin(), threads.end(), [](std::thread& x) {x.join(); });
	var endNow = std::chrono::system_clock::now();
	double delayms = std::chrono::duration_cast<std::chrono::milliseconds>(endNow - start).count();
	cout << "[" << identifier << "] Average Time=" << delayms / iterations <<
		"ms | Throughput=" <<
		(1000 * iterations / delayms) * FLOAT_COUNT * 4 / 1024 / 1024 / 1024 <<
		"GB/s" <<
		endl;
}

int main()
{
	int numProcs = PROC_CNT;
	vector<INAKernel> kernels;
	vector<void*> args;
#pragma region RunNaivePass
	const char* NaivePass = "NAIVE FLAT PASS";
	kernels.clear();
	args.clear();
	for (int i = 0; i < numProcs; i++)
	{
		kernels.push_back(RunNaivePass);
		args.push_back(NULL);
	}
	TestIterator(ITERATION, kernels, args,string(NaivePass));
#pragma endregion

#pragma region RunReductionTree
	kernels.clear();
	args.clear();
	const char* ReductionTree = "NAIVE REDUCTION TREE BASE-2";
	ReductionTreeArgs redArgs;
	redArgs.Base = 2;
	ScratchReductionTree = new real_t*[PROC_CNT];
	//use no more than this.
	for (int i = 0; i < numProcs; i++)
	{
		ScratchReductionTree[i] = new real_t[FLOAT_COUNT];
		args.push_back(&redArgs);
		kernels.push_back(RunNaiveReductionTreePass);
	}
	TestIterator(ITERATION, kernels, args, string(ReductionTree));
	for (int i = 0; i < numProcs; i++)
	{
		delete ScratchReductionTree[i];
	}
#pragma endregion
}