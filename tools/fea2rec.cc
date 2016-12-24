/*!
 * \file fea2rec.cc
 *  Image List Format: unique-image-index label[s] path-to-image
 * \sa dmlc/recordio.h
 */
#include <cctype>
#include <cstring>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <dmlc/logging.h>
#include <dmlc/recordio.h>
#include "../src/io/image_recordio.h"
#include <random>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: <image.lst> <feature.bin> <output.rec> [additional parameters in form key=value]\n"\
           "Possible additional parameters:\n"\
           "\tfeature_dim=dim[default=1024] specify the dimension of the feature, by default set to 1024\n"\ 
           "\tlabel_width=WIDTH[default=1] specify the label_width in the list, by default set to 1\n"\
           "\tnsplit=NSPLIT[default=1] used for part generation, logically split the image.list to NSPLIT parts by position\n"\
           "\tpart=PART[default=0] used for part generation, pack the images from the specific part in image.list\n");
    return 0;
  }
  int feature_dim = 1024;
  int label_width = 1;
  int nsplit = 1;
  int partid = 0;
  
  for (int i = 4; i < argc; ++i) {
    char key[128], val[128];
    int effct_len = 0;
    
#ifdef _MSC_VER
    effct_len = sscanf_s(argv[i], "%[^=]=%s", key, sizeof(key), val, sizeof(val));
#else
    effct_len = sscanf(argv[i], "%[^=]=%s", key, val);
#endif
    
    if (effct_len == 2) {
      if (!strcmp(key, "feature_dim")) feature_dim = atoi(val);
      if (!strcmp(key, "label_width")) label_width = atoi(val);
      if (!strcmp(key, "nsplit")) nsplit = atoi(val);
      if (!strcmp(key, "part")) partid = atoi(val);
    }
  }

  if (label_width != 1) {
    LOG(FATAL) << "label_width must be 1";
  }
  

  using namespace dmlc;
  std::string feature_path = argv[2];
  mxnet::io::ImageRecordIO rec;
  size_t imcnt = 0;
  double tstart = dmlc::GetTime();
  dmlc::InputSplit *flist = dmlc::InputSplit::
      Create(argv[1], partid, nsplit, "text");
  std::ostringstream os;
  if (nsplit == 1) {
    os << argv[3];
  } else {
    os << argv[3] << ".part" << std::setw(3) << std::setfill('0') << partid;
  }
  LOG(INFO) << "Write to output: " << os.str();
  dmlc::Stream *fo = dmlc::Stream::Create(os.str().c_str(), "w");
  LOG(INFO) << "Output: " << os.str();
  dmlc::RecordIOWriter writer(fo);
  std::string blob;
  size_t feature_buf_size = feature_dim * sizeof(double);
  char* feature_buf = (char*)malloc(feature_buf_size);
  
  dmlc::InputSplit::Blob line;
  std::vector<float> label_buf(label_width, 0.f);


  // use "r" is equal to rb in dmlc::Stream
  dmlc::Stream *fi = dmlc::Stream::Create(feature_path.c_str(), "r");
  if (!fi)  return -1;

  while (flist->NextRecord(&line)) {
    std::string sline(static_cast<char*>(line.dptr), line.size);
    std::istringstream is(sline);
    if (!(is >> rec.header.image_id[0] >> rec.header.label)) continue;
    label_buf[0] = rec.header.label;
    for (int k = 1; k < label_width; ++k) {
      CHECK(is >> label_buf[k])
          << "Invalid ImageList, did you provide the correct label_width?";
    }
    rec.SaveHeader(&blob);

    //memset(feature_buf,0,feature_buf_size);
    size_t nread = fi->Read(feature_buf, feature_buf_size);
    if (nread != feature_buf_size) break;

    size_t bsize = blob.size();
    blob.resize(bsize + feature_buf_size);
    memcpy(BeginPtr(blob) + bsize, feature_buf, feature_buf_size);
    writer.WriteRecord(BeginPtr(blob), blob.size());

    // write header
    ++imcnt;
    if (imcnt % 1000 == 0) {
      LOG(INFO) << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
    }
  }
  LOG(INFO) << "Total: " << imcnt << " images processed, " << GetTime() - tstart << " sec elapsed";
  delete fo;
  delete flist;
  if (feature_buf)
      free(feature_buf);
  return 0;
}
