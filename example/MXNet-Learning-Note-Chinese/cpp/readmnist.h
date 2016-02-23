#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <mxnet/ndarray.h>
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/symbolic.h>
#include <mxnet/optimizer.h>
using namespace std;
int ReverseInt(int i);
int read_Mnist(string filename, vector<vector<float> > &vec);
void read_Mnist_Label(string filename, vector<float> &vecl);
int getplane(vector<float> &vec, vector<float> &vecl);
size_t Getdata(vector<float> &data, vector<float> &label);