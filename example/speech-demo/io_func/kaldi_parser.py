from __future__ import print_function
import struct
import numpy as num
import sys

class KaldiParser(object):

    NO_OPEN_BRACKET = "found > before <"
    ERR_NO_CLOSE_BRACKET = "reached eof before >"
    ERR_BYTES_BEFORE_TOKEN = "found bytes before <"
    NO_SPACE_AFTER = "missing space after >"

    def __init__(self, f):
        self.f = f
        self.binary = self.f.read(2) == '\0B'
        assert(self.binary), "text format not supported yet"
        if not self.binary:
            self.f.seek(0, 0)

    def is_binary(self):
        return self.binary

    def try_next_token(self):
        pos = self.f.tell()
        err, tok = self.next_token()
        if err is not None:
            self.f.seek(pos, 0)
            print(err, tok)
            return None
        return tok.lower()

    def next_token(self):
        # keep reading until you get a > or at end of file (return None)
        # consume the space
        # return substring from < to >
        # if things before < are not space, return error
        buf = ""
        while True:
            b = self.f.read(1)
            if b is None:
                return KaldiParser.ERR_NO_CLOSE_BRACKET, None
            buf += b
            if b == ">":
                break

        try:
            start = buf.index("<")
        except ValueError:
            return KaldiParser.NO_OPEN_BRACKET, None

        b = self.f.read(1)
        if not (b == " " or b is None):
            return KaldiParser.NO_SPACE_AFTER, buf[start:]

        if start != 0:
            return KaldiParser.ERR_BYTES_BEFORE_TOKEN, buf[start:]

        return None, buf

    def read_space(self):
        b = self.f.read(1)
        assert(b == " " or b is None)

    # http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    def read_basic_type(self, type):
        if self.binary:
            size = num.fromfile(self.f, dtype=num.dtype("i1"), count=1)[0]

            if type == "int":
                dtype = "<i4"
                dsize = 4
            elif type == "float":
                dtype = "<f4"
                dsize = 4
            elif type == "char":
                dtype = 'a'
                dsize = 1
            else:
                print("unrecognized type")
                return None

            assert(size == dsize)
            n = num.fromfile(self.f, dtype=num.dtype(dtype), count=1)
            return n[0]

        else:
            assert(False), "not supported yet"

    def read_matrix(self):
        mode = self.f.read(2)
        #print mode
        assert(mode == 'FM')
        self.read_space()

        rows = self.read_basic_type("int")
        #print "rows", rows
        cols = self.read_basic_type("int")
        #print "cols", cols

        n = num.fromfile(self.f, dtype=num.dtype("<f4"), count=rows * cols)
        n = n.reshape((rows, cols))

        #print n[0][0]
        #print "-----------"
        return n

    def read_vector(self):
        mode = self.f.read(2)
        #print mode
        assert(mode == 'FV')
        self.read_space()

        length = self.read_basic_type("int")
        #print "length", length

        n = num.fromfile(self.f, dtype=num.dtype("<f4"), count=length)
        #print n[0]
        #print "-----------"
        return n

def fileIsBinary(filename):
    f = open(filename, "rb")
    binary = (f.read(2) == '\0B')
    f.seek(0, 0)
    return binary

def file2nnet_binary(filename):
    f = open(filename, "rb")
    parser = KaldiParser(f)

    net = []
    layer = None
    while True:
        tok = parser.try_next_token()
        if tok is None:
            print("error")
            break
        if tok == "<nnet>":
            continue
        elif tok == "<affinetransform>":
            if layer is not None:
                net += [layer]
            layer = {}
            layer["outdim"] = parser.read_basic_type("int")
            layer["indim"] = parser.read_basic_type("int")
        elif tok == "<learnratecoef>":
            parser.read_basic_type("float")
        elif tok == "<biaslearnratecoef>":
            parser.read_basic_type("float")
        elif tok == "<maxnorm>":
            parser.read_basic_type("float")
            layer["weights"] = parser.read_matrix().transpose()        # kaldi writes the transpose!!!!
            layer["bias"] = parser.read_vector()
        elif tok == "<sigmoid>" or tok == "<softmax>":
            layer["type"] = tok[1:-1]
            outdim1 = parser.read_basic_type("int")
            outdim2 = parser.read_basic_type("int")
            assert(outdim1 == outdim2 and outdim2 == layer["outdim"])
        elif tok == "</nnet>":
            #print "Done!"
            break
        else:
            print("unrecognized token", tok)
            break

    if layer is not None:
        net += [layer]

    #for layer in net:
    #    print layer.keys()

    return net

if __name__ == '__main__':
    filename = "exp/dnn4_pretrain-dbn_dnn/nnet_6.dbn_dnn.init"
    #filename = "/usr/users/leoliu/s5/exp/dnn4_pretrain-dbn_dnn/final.feature_transform"
    print(filename)

    print("isBinary:", fileIsBinary(filename))
    a = file2nnet_binary(filename)



    """
    while True:
        err, tok = parser.next_token()
        if err != KaldiParser.NO_SPACE_AFTER and tok is not None:
            print(err, tok)
    """

"""
        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        if maxout:
            fout.write('<maxout> ' + str(int(layers[i + 1])) + ' ' + str(output_size) + '\n')
        else:
            fout.write('<sigmoid> ' + str(output_size) + ' ' + str(output_size) + '\n')
"""