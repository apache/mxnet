#include <cmath>
#include <iostream>
#include "omp.h"
#include <chrono>

void quantize_2bit(float* data, float* res, float* compr, long long int size){
    #pragma omp parallel for
    for(long long int i=0; i<size/16; i++) {
        float* compr_block = compr + i;
        *compr_block = 0;
        int s=i<<4, e=s+16;
        char* block_ptr = reinterpret_cast<char*>(compr_block);
        const int posbits[] = {0xc0, 0x30, 0x0c, 0x03};
        const int negbits[] = {0x80, 0x20, 0x08, 0x02};
        char* curr_byte = block_ptr;
        for(int i=s; i<e && i<size; i++){
            curr_byte += (i-s) & 3;
            res[i] += data[i];
            if (res[i] >= 0.5) {
                res[i] -= 0.5;
                *curr_byte |= posbits[i&3];
            }
            else if(res[i] <= -0.5) {
                res[i] += 0.5;
                *curr_byte |= negbits[i&3];
            }
        }
    }
}

int main() {
    std::cout<<"openmp max threads are "<<omp_get_max_threads()<<std::endl;
    const long long int size = 250000000;
    float* data = new float[size];
    float* compr = new float[size/16];
    float* residual = new float[size];
    
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for(long long int i=0; i<size; ++i) { 
        data[i] = 0;
        residual[i] = 0;
    }
    /*#pragma omp parallel for
    for(long long int i=0;i<size/16;i++){
        compr[i] = 0;
    } */
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "time for " <<size<< " is " << dur <<". Speed is "<< size*1000*4 / (dur*1000*1000*1000) <<" GBytes per sec"<< std::endl;  
   
    t1 = std::chrono::high_resolution_clock::now();
    quantize_2bit(data, residual, compr, size); 
    t2 = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout<< "time for quantizing "<<size<<" is "<<dur<<std::endl;
    return 0; 
}
