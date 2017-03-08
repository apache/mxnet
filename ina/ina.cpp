#include <stdio.h>
#include <unistd.h>
#include <functional>
#include <iostream>
#include <vector>
#include <thread>
#include <experimental/barrier>
#include <chrono>
#include <string>
using namespace std;

#define FLOAT_COUNT 104857600 //400MB
#define ITERATION 1000
#define COPIES 8
#define READ_ONLY_SAL
#define READ_WRITE_SAL
#define __OUT__
#define ALIGNED(x) __attribute__((aligned(x)))
#define CACHE_LINE_SIZE 64

typedef real_t float

READ_ONLY_SAL Inputs real_t[FLOAT_COUNT] ALIGNED(CACHE_LINE_SIZE); //put garbage in
READ_WRITE_SAL Outputs real_t[FLOAT_COUNT] ALIGNED(CACHE_LINE_SIZE);

void GetRange(int keyCnt, int copyCnt, int procCnt, int id, int& __OUT__ start, int& __OUT__ end)
{
    //key >> procCnt
    int procPerGroup = procCnt / 
}

void RunNaivePass(int proc)
{

}

void RunNaiveReductionTreePass(int proc)
{

}

void TestIterator(int iterations, vector<function<void(int)>> kernels, vector<int> cpuid2KernelMap, string identifier)
{
    vector<thread> threads(cpuid2KernelMap.size());
    barrier task_barrier(cpuid2KernelMap.size());
    for(int i = 0; i < cpuid2KernelMap.size(); i++)
    {
        threads[i] = thread(bind(
            [&](int total, int id, function<void(int)> kernel)
            {
                for(int it = 0; it < total; it++)
                {
                    kernel(id);
                    barrier.arrive_and_wait();
                }
            },iterations,i,kernels[cpuid2KernelMap[i]]
        ));
    }
    auto start = std::chrono::system_clock::now();
    for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
    auto endNow = std::chrono::system_clock::now();
    double delayms = std::chrono::duration_cast<std::chrono::microseconds>(endNow - start).count();
    cout<<"["<<identifier<<"] Average Time="<<delayms / iterations << 
          " | Throughput="<< 
          (1000 * iterations / delayms) * FLOAT_COUNT * 4 / 1024 / 1024 / 1024 <<
          "GB/s"<<
          endl;
}

int main()
{
    int numProcs = thread::hardware_concurrency();
    TestIterator(ITERATION, ~)
}