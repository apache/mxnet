#ifndef PTIMERUTIL_H_
#define PTIMERUTIL_H
#include <iostream>
#include <string>
#include <chrono>
#include <unordered_map>

namespace PTimerUtil {
  using namespace std;
  struct tval {
    chrono::system_clock::time_point start;
    chrono::duration<double> csum;
    chrono::duration<double> sum;
    long loops;
    long cloops;
  };
  static unordered_map<string, tval> tmap; 
  
  inline void tstart(string name, int printloops=1) {
    tval e;
    if (tmap.find(name) == tmap.end()) {
      e.sum = e.csum = chrono::duration<double>(0);
      e.loops = printloops;
      e.cloops = 0;
    }
    else {
      e = tmap[name];
    }
    if (e.loops < 1) return;
    e.start =  chrono::high_resolution_clock::now(); 
    tmap[name] = e;
  }

  inline void tprint(string name) {
    if (tmap.find(name) == tmap.end()) return;
    auto d = tmap[name];
    if (d.loops < 1) return;
    d.cloops++;
    d.csum += (chrono::high_resolution_clock::now()-d.start);
    if (d.cloops % d.loops == 0) {
      auto t = d.csum/d.loops;
      d.sum += d.csum;
      d.csum = chrono::duration<double>(0);
      cout << name << ": Current "<<t.count() << " Total " <<d.sum.count() <<" Iter "<<d.cloops<<endl;
    } 
    tmap[name] = d;
  }
}
#endif  // PTIMERUTIL_H_
