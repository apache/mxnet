import os
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np
import multiprocessing as mp
import mxnet as mx
from mxnet import gluon as gl
from mxnet.gluon.data.vision import transforms

if __name__ == '__main__':
	M = 24
	BS = 100

	dataset = gl.data.vision.ImageFolderDataset('../256_ObjectCategories')
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.RandomBrightness(1.0),
									transforms.RandomContrast(1.0),
									transforms.RandomSaturation(1.0),
									transforms.Normalize([0, 0, 0], [1, 1, 1])])
	dataset = dataset.transform_first(lambda x: transform(mx.image.center_crop(x, (224, 224))[0]))
	data_loader = gl.data.DataLoader(dataset, BS, shuffle=True, num_workers=M)

	N = len(dataset)

	iterator = iter(data_loader)

	tic = time.time()

	for data, label in iterator:
		data.wait_to_read()
		print(data.shape)

	print(N/(time.time() - tic))
