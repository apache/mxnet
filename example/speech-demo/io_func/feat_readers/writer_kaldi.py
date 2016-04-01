import sys
import numpy
import struct

# Functions to read and write Kaldi binary-formatted .scp and .ark

class KaldiWriteOut(object):

    def __init__(self, scp_path, ark_path):

        self.ark_path = ark_path
        self.scp_path = scp_path
        self.out_ark = None
        self.out_scp = None
        if sys.byteorder != 'little':
            raise Exception("output file needs to be little endian")

    def open(self):
        self.out_ark = open(self.ark_path,"w")
        self.out_scp = open(self.scp_path,"w")

    def write(self, uttID, data):
        assert data.dtype == numpy.float32

        self.out_ark.write(uttID + ' ')
        start_offset = self.out_ark.tell()

        # write out ark 
        num_row, num_col = data.shape
        self.out_ark.write('\0B')
        self.out_ark.write('FM ')
        self.out_ark.write(chr(4))
        self.out_ark.write(struct.pack('i', num_row))
        self.out_ark.write(chr(4))
        self.out_ark.write(struct.pack('i', num_col))
        data.tofile(self.out_ark)

        # write out scp
        scp_out = uttID + ' ' + self.ark_path + ':' + str(start_offset)
        self.out_scp.write(scp_out + '\n')

    def close(self):
        self.out_ark.close()
        self.out_scp.close()
