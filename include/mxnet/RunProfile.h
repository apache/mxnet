#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <thread>

class RuntimeProfile
{
    public:
    static RuntimeProfile* Get() {
        static RuntimeProfile e; return &e;
    }
    int GetRuntime(std::string& oprName)
    {
        return store[oprName];
    }
    private:
    RuntimeProfile()
    {
        std::ifstream infile("runProfile.txt");
        std::string name;
        int time;
        while(infile>> name>>time)
        {
            store[name] = time;
        }
    }
    std::unordered_map<std::string,int> store;
};

namespace mxnet{
static inline void emptyFunc1(mxnet::RunContext cntx)
{
    int sleepTime = 0;
   sleepTime = RuntimeProfile::Get()->GetRuntime(cntx.oprName);
   std::this_thread::sleep_for (std::chrono::milliseconds(sleepTime));
}
static void emptyFunc2(mxnet::RunContext cntx, mxnet::engine::CallbackOnComplete cb)
{
   //cntx.
   //RuntimeProfile::Get()->GetRuntime()
   emptyFunc1(cntx);
   //if(cb!=NULL)
   cb();
}
}