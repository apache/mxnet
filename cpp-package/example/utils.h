#include <map>
#include <string>
#include <fstream>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;

bool isFileExists(const string &filename) {
  ifstream fhandle(filename.c_str());
  return fhandle.good();
}

bool check_datafiles(vector<string> &data_files) {
  for (size_t index=0; index < data_files.size(); index++) {
    if (!(isFileExists(data_files[index]))) {
      LG << "Error: File does not exist: "<< data_files[index];
      return false;
    }
  }
  return true;
  }

bool setDataIter(MXDataIter &iter , string useType, vector<string> &data_files, int batch_size)
{
    if(!check_datafiles(data_files))
        return false;

    if(useType ==  "Train") {
                iter.SetParam("image", data_files[0]);
                iter.SetParam("label", data_files[1]);
                iter.SetParam("batch_size", batch_size);
                iter.SetParam("shuffle", 1);
                iter.SetParam("flat", 0);
    }
    else if(useType == "Label") {
                iter.SetParam("image", data_files[2]);
                iter.SetParam("label", data_files[3]);
    }

    iter.CreateDataIter();
    return true;
}