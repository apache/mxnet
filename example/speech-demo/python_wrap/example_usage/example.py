import ctypes
import numpy

c_float_ptr = ctypes.POINTER(ctypes.c_float)
c_int_ptr = ctypes.POINTER(ctypes.c_int)
c_void_p = ctypes.c_void_p
c_int = ctypes.c_int
c_char_p = ctypes.c_char_p
c_float = ctypes.c_float

kaldi = ctypes.cdll.LoadLibrary("libkaldi-python-wrap.so")  # this needs to be in LD_LIBRARY_PATH

def decl(f, restype, argtypes):
    f.restype = restype
    if argtypes is not None and len(argtypes) != 0:
        f.argtypes = argtypes

decl(kaldi.Foo_new, c_void_p, [])
decl(kaldi.Foo_bar, None, [c_void_p])
decl(kaldi.Foo_getx, c_float_ptr, [c_void_p])
decl(kaldi.Foo_sizex, c_int, [c_void_p])

decl(kaldi.SBFMReader_new,          c_void_p,   [])
decl(kaldi.SBFMReader_new_char,     c_void_p,   [c_char_p])
decl(kaldi.SBFMReader_Open,         c_int,      [c_void_p, c_char_p])
decl(kaldi.SBFMReader_Done,         c_int,      [c_void_p])
decl(kaldi.SBFMReader_Key,          c_char_p,   [c_void_p])
decl(kaldi.SBFMReader_FreeCurrent,  None,       [c_void_p])
decl(kaldi.SBFMReader_Value,        c_void_p,   [c_void_p])
decl(kaldi.SBFMReader_Next,         None,       [c_void_p])
decl(kaldi.SBFMReader_IsOpen,       c_int,      [c_void_p])
decl(kaldi.SBFMReader_Close,        c_int,      [c_void_p])
decl(kaldi.SBFMReader_Delete,       None,       [c_void_p])

decl(kaldi.MatrixF_NumRows,     c_int,       [c_void_p])
decl(kaldi.MatrixF_NumCols,     c_int,       [c_void_p])
decl(kaldi.MatrixF_Stride,      c_int,       [c_void_p])
decl(kaldi.MatrixF_cpy_to_ptr,  None,        [c_void_p, c_float_ptr, c_int])
decl(kaldi.MatrixF_SizeInBytes, c_int,       [c_void_p])
decl(kaldi.MatrixF_Data,        c_float_ptr, [c_void_p])

if __name__ == "__main__":
    print "-------- Foo class example --------"
    a = kaldi.Foo_new()
    print "Calling Foo_bar(): ",
    kaldi.Foo_bar(a)
    print
    print "Result of Foo_getx(): ", kaldi.Foo_getx(a)
    print "Result of Foo_sizex(): ", kaldi.Foo_sizex(a)

    print
    print "-------- Kaldi SBFMReader and MatrixF class example --------"

    reader = kaldi.SBFMReader_new_char("scp:data.scp")
    
    # data.scp has exactly one utterance, assert it's there
    assert(not kaldi.SBFMReader_Done(reader))

    utt_id = kaldi.SBFMReader_Key(reader)

    feat_value = kaldi.SBFMReader_Value(reader)
    feat_rows = kaldi.MatrixF_NumRows(feat_value)
    feat_cols = kaldi.MatrixF_NumCols(feat_value)
    feat_data = kaldi.MatrixF_Data(feat_value)
    
    # never use numpy.ndarray(buf=) or numpy.ctypeslib.as_array
    # because you don't know if Python or C owns buffer
    # (even if you numpy.copy() resulting array)
    # http://stackoverflow.com/questions/4355524/getting-data-from-ctypes-array-into-numpy
    #
    # Can't use memmove/memcpy because arrays are strided
    # Use cpy_to_ptr
    feats = numpy.empty((feat_rows,feat_cols), dtype=numpy.float32)

    # MUST: cast Python int to pointer, otherwise C interprets as 32-bit
    # if you print the pointer value before casting, you might see weird value before seg fault
    # casting fixes that
    feats_numpy_ptr = ctypes.cast(feats.ctypes.data, c_float_ptr)
    kaldi.MatrixF_cpy_to_ptr(feat_value, feats_numpy_ptr, feats.strides[0]/4)

    print "Read utterance:"
    print "  ID: ", utt_id
    print "  Rows: ", feat_rows
    print "  Cols: ", feat_cols
    print "  Value: ", feat_data
    print feats
    print "  This should match data.txt"

    # assert no more utterances left
    kaldi.SBFMReader_Next(reader)
    assert(kaldi.SBFMReader_Done(reader))

    kaldi.SBFMReader_Delete(reader)
    
