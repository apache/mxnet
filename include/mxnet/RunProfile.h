#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
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
