try:
    import caffe
    from caffe.proto import caffe_pb2
    use_caffe = True
except ImportError:
    try:
        import caffe_pb2
    except ImportError:
        raise ImportError('You used to compile with protoc --python_out=./ ./caffe.proto')
    use_caffe = False
