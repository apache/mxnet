#ifndef MXNET_UTILS_DECODER_H_
#define MXNET_UTILS_DECODER_H_

#include <vector>
#if MXNET_USE_OPENCV_DECODER == 0
  #include <jpeglib.h>
  #include <setjmp.h>
  #include <jerror.h>
#endif
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#if MXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif

namespace cxxnet {
namespace utils {

#if MXNET_USE_OPENCV_DECODER == 0
struct JpegDecoder {
public:
  JpegDecoder(void) {
    cinfo.err = jpeg_std_error(&jerr.base);
    jerr.base.error_exit = jerror_exit;
    jerr.base.output_message = joutput_message;
    jpeg_create_decompress(&cinfo);
  }
  // destructor
  ~JpegDecoder(void) {
    jpeg_destroy_decompress(&cinfo);
  }

  inline void Decode(unsigned char *ptr, size_t sz,
                     mshadow::TensorContainer<cpu, 3, unsigned char> *p_data) {
    if(setjmp(jerr.jmp)) {
      jpeg_destroy_decompress(&cinfo);
      dmlc::Error("Libjpeg fail to decode");
    }
    this->jpeg_mem_src(&cinfo, ptr, sz);
    CHECK(jpeg_read_header(&cinfo, TRUE) == JPEG_HEADER_OK) << "libjpeg: failed to decode";
    CHECK(jpeg_start_decompress(&cinfo) == true) << "libjpeg: failed to decode";
    p_data->Resize(mshadow::Shape3(cinfo.output_height, cinfo.output_width, cinfo.output_components));
    JSAMPROW jptr = &((*p_data)[0][0][0]);
    while (cinfo.output_scanline < cinfo.output_height) {
      CHECK(jpeg_read_scanlines(&cinfo, &jptr, 1) == true) << "libjpeg: failed to decode";
      jptr += cinfo.output_width * cinfo.output_components;
    }
    CHECK(jpeg_finish_decompress(&cinfo) == true) << "libjpeg: failed to decode");
  }
private:
  struct jerror_mgr {
    jpeg_error_mgr base;
    jmp_buf jmp;
  };

  METHODDEF(void) jerror_exit(j_common_ptr jinfo) {
    jerror_mgr* err = (jerror_mgr*)jinfo->err;
    longjmp(err->jmp, 1);
  }

  METHODDEF(void) joutput_message(j_common_ptr) {}

  static boolean mem_fill_input_buffer_ (j_decompress_ptr cinfo) {
    dmlc::Error("JpegDecoder: bad jpeg image");
    return true;
  }

  static void mem_skip_input_data_ (j_decompress_ptr cinfo, long num_bytes_) {
    jpeg_source_mgr *src = cinfo->src;
    size_t num_bytes = static_cast<size_t>(num_bytes_);
    if (num_bytes > 0) {
      src->next_input_byte += num_bytes;
      CHECK(src->bytes_in_buffer >= num_bytes) << "fail to decode";
      src->bytes_in_buffer -= num_bytes;
    } else {
      dmlc::Error("JpegDecoder: bad jpeg image");

    }
  }

  static void mem_term_source_ (j_decompress_ptr cinfo) {}
  static void mem_init_source_ (j_decompress_ptr cinfo) {}
  static boolean jpeg_resync_to_restart_(j_decompress_ptr cinfo, int desired) {
    dmlc::Error("JpegDecoder: bad jpeg image");
    return true;
  }
  void jpeg_mem_src (j_decompress_ptr cinfo, void* buffer, long nbytes) {
    src.init_source = mem_init_source_;
    src.fill_input_buffer = mem_fill_input_buffer_;
    src.skip_input_data = mem_skip_input_data_;
    src.resync_to_restart = jpeg_resync_to_restart_;
    src.term_source = mem_term_source_;
    src.bytes_in_buffer = nbytes;
    src.next_input_byte = static_cast<JOCTET*>(buffer);
    cinfo->src = &src;
  }

private:
  jpeg_decompress_struct cinfo;
  jpeg_source_mgr src;
  jerror_mgr jerr;
};
#endif

#if MXNET_USE_OPENCV
struct OpenCVDecoder {
  void Decode(unsigned char *ptr, size_t sz, mshadow::TensorContainer<cpu, 3, unsigned char> *p_data) {
    cv::Mat buf(1, sz, CV_8U, ptr);
    cv::Mat res = cv::imdecode(buf, 1);
    CHECK(res.data != NULL) << "decoding fail";
    p_data->Resize(mshadow::Shape3(res.rows, res.cols, 3));
    for (int y = 0; y < res.rows; ++y) {
      for (int x = 0; x < res.cols; ++x) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(y, x);
        // store in RGB order
        (*p_data)[y][x][2] = bgr[0];
        (*p_data)[y][x][1] = bgr[1];
        (*p_data)[y][x][0] = bgr[2];
      }
    }
    res.release();
  }
};
#endif
} // namespace utils
} // namespace mxnet

#endif // DECODER_H
