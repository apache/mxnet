import numpy as np
import mxnet as mx

def test_impulse_response(dil=(1,1), kernel_shape=(3,3), verbose=False):
    # Input for spike response
    spike_imgs = np.zeros(shape=(1,1,33,33), dtype=np.float32)
    spike_imgs[0,0,16,16] = 1.0
    spike_img = mx.nd.array(spike_imgs)
    spike_img2 = mx.nd.array(spike_imgs)


    kernel_weights = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)
    kernel_weights2 = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)

    kernel = mx.symbol.Variable('kernel')
    in_img = mx.symbol.Variable('input')
    net = mx.symbol.Convolution(in_img, num_filter=1,kernel=kernel_shape, dilate=dil, no_bias="true", name='test_convolution')
    net.list_arguments()
    be = net.bind(mx.cpu(), args={ 'input' : spike_img, 'test_convolution_weight' : kernel_weights},
                args_grad={'input' : spike_img2, 'test_convolution_weight' : kernel_weights2 } )
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    ndo = be.outputs[0]
    
    out_grads = np.zeros(shape=be.outputs[0].shape, dtype=np.float32)
    out_grads[0,0, 16,16] = 1.0
    out_grad = mx.nd.array(out_grads)
    be.backward([out_grad])
    vgrad = be.grad_arrays[0].asnumpy()
    out = out_o.reshape((out_o.shape[2],out_o.shape[3]))
    nzx,nzy = np.nonzero(out)
    assert(np.sum(out)==np.prod(kernel_shape))
    assert(np.sum(vgrad)==np.prod(kernel_shape))

    if (verbose):
	    print "Output Shape = %d,%d" % (be.outputs[0].shape[2],be.outputs[0].shape[3] )
	    print "Impulse Response for convolution with kernel shape %d, %d, dilate=%d,%d - Output Size: %d, %d" % (kernel_shape[0],kernel_shape[1], dil[0], dil[1], out.shape[0], out.shape[1])
	    for i in range(len(nzx)):
        	print "(%d,%d)=%f" % (nzx[i],nzy[i], out[nzx[i], nzy[i]])
    if (verbose):
	    np.set_printoptions(threshold=1000000)
	    print "Input Inverse Impulse Response for convolution with kernel shape %d, %d, dilate=%d,%d - Output Size: %d, %d" % (kernel_shape[0],kernel_shape[1], dil[0], dil[1], vgrad.shape[2], vgrad.shape[3])
	    for i in range(vgrad.shape[2]):
		for j in range(vgrad.shape[3]):
			if (vgrad[0,0,i,j]!=0.0):
				print "(%d,%d)=%f" % (i,j,vgrad[0,0,i,j] )

    # Now check whether the input gradient was computed correctly
    input_grad = mx.nd.array(vgrad)

    be = net.bind(mx.cpu(), args={ 'input' : input_grad, 'test_convolution_weight' : kernel_weights})
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    assert(out_o[0,0,16,16]==np.prod(kernel_shape))

    rnd_kernel_s = np.random.uniform(low=0.0, high=1.0, size=tuple([1,1]+list(kernel_shape))).astype(np.float32)
    impulse_error = mx.nd.array(out_o/np.sum(out_o)) # This should be 1.0 at [0,0,16,16]
    rnd_kernel = mx.nd.array(rnd_kernel_s)

    rnd_kernel2 = mx.nd.array(rnd_kernel_s)
    white_in = mx.nd.ones(shape=(1,1,33,33))
    white_in2 = mx.nd.ones(shape=(1,1,33,33))

    be = net.bind(mx.cpu(), args={ 'input' : white_in, 'test_convolution_weight' : rnd_kernel},
                args_grad={'input' : white_in2, 'test_convolution_weight' : rnd_kernel2 } )

    be.forward(True)
    be.backward([impulse_error])
    out_orig = be.outputs[0].asnumpy()
    kernel_gradient = be.grad_arrays[1].asnumpy()
    
    dkernel = mx.nd.array(rnd_kernel_s + kernel_gradient)

    be = net.bind(mx.cpu(), args={ 'input' : white_in, 'test_convolution_weight' : dkernel})

    be.forward(True)
    out = be.outputs[0].asnumpy()
    # Now do a simple check of the kernel gradient
    assert(out[0,0,16,16] - np.sum(kernel_gradient) - out_orig[0,0,16,16] < 0.001)
    

if __name__=='__main__':  
	test_impulse_response((1,1), kernel_shape=(3,3), verbose=False)
	test_impulse_response((2,2), kernel_shape=(3,3), verbose=False)
	test_impulse_response((3,3), kernel_shape=(3,3), verbose=False)
	test_impulse_response((2,2), kernel_shape=(4,4), verbose=False)
	test_impulse_response((2,2), kernel_shape=(2,3), verbose=False)
	test_impulse_response((1,1), kernel_shape=(2,3), verbose=False)
	test_impulse_response((3,3), kernel_shape=(2,3), verbose=False)
	test_impulse_response((3,3), kernel_shape=(3,2), verbose=False)
	test_impulse_response((1,1), kernel_shape=(1,1), verbose=False)


