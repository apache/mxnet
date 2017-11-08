#include "../operator/mxnet_op.h"
namespace mxnet {
  namespace kvstore{
    struct quantize_2bit {
      MSHADOW_XINLINE static void Map(int out_block_id,
                                      int original_size,
                                      float *out,
                                      float *grad,
                                      float *residual,
                                      const float neg_threshold,
                                      const float pos_threshold) {
        float *compr_block = out + out_block_id;
        // init to 0
        *compr_block = 0;
        // start and end are indices in original grad array
        int start = out_block_id << 4;
        int end = start + 16; // <= original_size) ? start + 16 : original_size;
        char *block_ptr = reinterpret_cast < char * > (compr_block);
        const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
        const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};

        for (int i = start; i < end && i < original_size; i++) {
          // // adds 1 when i-start divisible by 4
          char *curr_byte = block_ptr + ((i - start) >> 2);
          residual[i] += grad[i];
          if (residual[i] >= pos_threshold) {
            residual[i] -= pos_threshold;
            // set data to 11
            *curr_byte |= posbits[(i & 3)];
//        std::cout<<"pos "<< std::to_string(i&3) << " " << std::bitset<8>(*curr_byte)<<std::endl;
          } else if (residual[i] <= neg_threshold) {
            residual[i] -= neg_threshold;
            // set data to 10
            *curr_byte |= negbits[(i & 3)];
//        std::cout<<"neg "<< std::to_string(i&3) << " " << std::bitset<8>(*curr_byte)<<std::endl;
          } else {
//        std::cout<<"0 "<< std::to_string(i&3) << " " << std::bitset<8>(*curr_byte)<<std::endl;
          }
        }
      }
    };

    template<typename xpu>
    void Quantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                                  const float threshold) {
      mxnet::op::mxnet_op::Kernel<mxnet::kvstore::quantize_2bit, xpu>::Launch(s,
                                                                              inputs[2].Size(), // compressed array size
                                                                              inputs[0].Size(),    // original size
                                                                              inputs[2].dptr<float>(),   // compressed array
                                                                              inputs[0].dptr<float>(),     // original array
                                                                              inputs[1].dptr<float>(),     // residual array
                                                                              -1 *
                                                                              threshold,            // negative threshold
                                                                              threshold);              // positive threshold
    }

    struct dequantize_2bit {
      // Decompress
      MSHADOW_XINLINE static void Map(int i,
                                      float *out,
                                      float *in,
                                      const float neg_threshold,
                                      const float pos_threshold) {

        float *outval = out + i;
        char *ch_ptr = reinterpret_cast<char *>(in + (i >> 4));

        ch_ptr += ((i & 15) >> 2);
        const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
        const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
        int col = i & 3;
        uint8_t mask = posbits[col];
        uint8_t negmask = negbits[col];
        uint8_t masked = *ch_ptr & mask;
        if (masked == mask) {
          *outval = pos_threshold;
        } // use posbits for mask as posbits are 11
          // compare with negbits
        else if (masked == negmask) {
          *outval = neg_threshold;
        } else {
          *outval = 0;
        }
      }
    };

    template<typename xpu>
    void Dequantize2BitKernelLaunch(mshadow::Stream<xpu> *s, const std::vector<mxnet::TBlob> &inputs,
                                    const float threshold) {

      mxnet::op::mxnet_op::Kernel<mxnet::kvstore::dequantize_2bit, xpu>::Launch(s, inputs[1].Size(),  // original size
                                                                                inputs[1].dptr<float>(),        // out array
                                                                                inputs[0].dptr<float>(),      // compressed array
                                                                                -1 *
                                                                                threshold,     // negative threshold
                                                                                threshold);       // positive threshold
    }


  }
}
