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
#define COPIES 8
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
extern bool __sync_bool_compare_and_swap(void* location, uint64_t old, uint64_t desired);
extern bool __sync_bool_compare_and_swap(void* location, int old, int desired);
#define BUSY_WAIT_NO_OP
#else
#define BUSY_WAIT_NO_OP asm("")
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
		} while (!__sync_bool_compare_and_swap(&ScratchNaive[i].backingStore, old.backingStore, target.backingStore));
		if (target.fv[1] == numProcs)
		{
			Outputs[i] = target.fv[0];
		}
	}
}
#pragma endregion

#pragma region MyRegion

class ReductionTreeScratch
{
public:
	real_t* Scratch;
	volatile int ReadyCursor ALIGNED(4);
	int sid;
} ALIGNED(CACHE_LINE_SIZE);
class ReductionTreeArgs
{
public:
	int Base;
};

ReductionTreeScratch* reductionTreeScratches;
void RunNaiveReductionTreePass(int proc, void* additionalArgs)
{
	var pArgs = (ReductionTreeArgs*)additionalArgs;
	var base = pArgs->Base;
	var level = (int)(log(proc + 1) / log(base));
	var totalProc = PROC_CNT;
	var maxLevel = (int)(log(totalProc - 1) / log(base));
	vector<ReductionTreeScratch*> sources;
	ReductionTreeScratch* dest;
	if (totalProc == proc + 1) return; //not sure what to do with this proc.
	ReductionTreeScratch inputFakeSource{ .Scratch = Inputs,.ReadyCursor = FLOAT_COUNT };
	ReductionTreeScratch outputFakeSink{ .Scratch = Outputs };
	//proc i read 2*i, 2*i+1
	if (level == 0)
	{
		//printf("proc %d is root layer.\n", proc);
		//the finally reduced 
		for (int i = 0; i < base; i++)
		{
			sources.push_back(&reductionTreeScratches[base*(proc + 1) + i - 1]);
			//reductionTreeScratches[base*(proc + 1) + i - 1].ReadyCursor = -1;
		}
		dest = &outputFakeSink;
		dest->ReadyCursor = -1;
	}
	else if (level == maxLevel)
	{
		//this is the leaf.
		//printf("proc %d is leaf layer.\n", proc);
		for (int i = 0; i < base; i++)
		{
			sources.push_back(&inputFakeSource);
		}
		dest = &reductionTreeScratches[proc];
		dest->ReadyCursor = -1;
	}
	else
	{
		//printf("proc %d is middle layer.\n", proc);
		//middle
		for (int i = 0; i < base; i++)
		{
			sources.push_back(&reductionTreeScratches[base*(proc + 1) + i - 1]);
		}
		dest = &reductionTreeScratches[proc];
		dest->ReadyCursor = -1;
	}

	//every thread does this:
	//probe to see if source has any data
	for (int i = 0; i < FLOAT_COUNT; i++)
	{
		//read from all sources
		var acc = 0;
		for (int s = 0; s < sources.size(); s++)
		{
			bool complained = false;
			while (sources[s]->ReadyCursor < i)
			{
				BUSY_WAIT_NO_OP;
				//if (complained == false)
				//{
				//	printf("Proc %d waiting for source sid=%d (%llx) whose redyCursor is %d\n", proc, sources[s]->sid, sources[s], sources[s]->ReadyCursor);
				//  complained = true;
				//}
			}
			//clear to read.
			acc += sources[s]->Scratch[i];
		}
		//write to destination.
		dest->Scratch[i] = acc;
		dest->ReadyCursor = i;
		//assert(dest->ReadyCursor == i);
		//__sync_bool_compare_and_swap((int*)&dest->ReadyCursor, dest->ReadyCursor, i);
		//printf("Proc %d advanced scratch sid=%d's (%llx) redyCursor to %d\n", proc, dest->sid, dest, dest->ReadyCursor);
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
				kernel(id, arg);
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
	goto REDUCTIONTREE;
	kernels.clear();
	args.clear();
	for (int i = 0; i < numProcs; i++)
	{
		kernels.push_back(RunNaivePass);
		args.push_back(NULL);
	}
	TestIterator(ITERATION, kernels, args, string(NaivePass));
#pragma endregion

#pragma region RunReductionTree
	REDUCTIONTREE :
				  kernels.clear();
				  args.clear();
				  const char* ReductionTree = "NAIVE REDUCTION TREE BASE-2";
				  ReductionTreeArgs redArgs;
				  redArgs.Base = 2;
				  reductionTreeScratches = new ReductionTreeScratch[PROC_CNT];
				  //use no more than this.
				  for (int i = 0; i < numProcs; i++)
				  {
					  reductionTreeScratches[i].Scratch = new real_t[FLOAT_COUNT];
					  reductionTreeScratches[i].sid = i;
					  args.push_back(&redArgs);
					  kernels.push_back(RunNaiveReductionTreePass);
				  }
				  TestIterator(ITERATION, kernels, args, string(ReductionTree));
				  for (int i = 0; i < numProcs; i++)
				  {
					  delete reductionTreeScratches[i].Scratch;
				  }
				  delete reductionTreeScratches;
#pragma endregion
}