#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
using namespace std;

class RuntimeProfile
{
    public:
    static RuntimeProfile* Get() {
        static RuntimeProfile e; return &e;
    }
    int GetRuntime(string& oprName)
    {
        return store[oprName];
    }
    private:
    RuntimeProfile()
    {
        ifstream infile("runProfile.txt");
        string name;
        int time;
        while(infile>> name>>time)
        {
            store[name] = time;
        }
    }
    unordered_map<string,int> store;
};