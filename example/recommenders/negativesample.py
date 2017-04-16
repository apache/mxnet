"""DataIter for negative sampling.
"""
import mxnet as mx
import numpy as np

class NegativeSamplingDataIter(mx.io.DataIter):
    """Wraps an existing DataIter to produce a new DataIter with negative samples.
    Assumes that all the relevant inputs are in data, not labels.
    Drops (replaces) any labels in the original DataIter.

    It only shuffles one of the input data columns, specified in the 
    constructor as shuffle_data_idx.  So if the original input data
    has three columns, ('item_ids', 'item_words', 'users') and you want
    to keep the two "item_*" together, then set `shuffle_data_idx=2`
    and `users` will be shuffled for the negative samples.

    Output batches will be larger than input batches by a factor
    of (1+sample_ratio)

    Negative samples are always drawn from the same minibatch.
    So they're effectively sampled according to the frequency at
    which they appear in the training data.  (Other reasonable
    samling strategies are not implemented here.)
    The shuffling is checked to ensure that a true positive sample
    isn't returned as a negative sample.
    """

    def __init__(self, source_dataiter, sample_ratio=1, shuffle_data_idx=1,
                 positive_label=1, negative_label=0):
        self._sourcedata = source_dataiter
        source_dataiter.reset()
        self.positive_label = positive_label  # output shapes = input shapes
        self.negative_label = negative_label
        self.shuffle_data_idx = shuffle_data_idx
        if sample_ratio == int(sample_ratio):
            self.sample_ratio = int(sample_ratio)
        else:
            raise ValueError("sample_ratio must be an integer, not %s" % sample_ratio)
        self._clear_queue()

        self.provide_data = source_dataiter.provide_data
        self.provide_label = source_dataiter.provide_label
        self.batch_size = source_dataiter.batch_size

    def _clear_queue(self):
        self._sampled_queue = []

    def _push_queue(self, data_list, labels):
        """Takes a list of numpy arrays for data, 
        and a numpy array for labels.
        Converts to minibatches and puts it on the queue.
        """
        num_minibatches = 1+self.sample_ratio
        total_size = len(labels)
        slice_size = total_size / num_minibatches
        def slicer(x, s):
            idx = range(s*slice_size, (s+1)*slice_size)
            return np.take(x,idx,0)

        for i in range(1+self.sample_ratio):
            nddata = [mx.nd.array(slicer(x,i)) for x in data_list]
            ndlabels = mx.nd.array(slicer(labels,i))
            batch = mx.io.DataBatch(nddata, [ndlabels], provide_data=self.provide_data,
                                    provide_label=self.provide_label)
            self._sampled_queue.append(batch)

    def next(self):
        if not self._sampled_queue:
            self._refill_queue()
        batch = self._sampled_queue.pop()
        return batch

    def reset(self):
        self._sourcedata.reset()
        self._clear_queue()

    def _shuffle_batch(self, data):
        # Takes a list of NDArrays.  Returns a shuffled version as numpy
        a = data[self.shuffle_data_idx].asnumpy()

        # Come up with a shuffled index
        batch_size = data[0].shape[0]
        si = np.arange(batch_size)
        np.random.shuffle(si)
        matches = (si == np.arange(batch_size)) # everywhere it didn't shuffle
        si -= matches  # Shifts down by 1 when true, ensuring it differs
        # Note shifting down by 1 works in python because -1 is a valid index.
        #Q: Is this shifting introducing bias?

        # Shuffle the data with the shuffle index
        shuf_a = np.take(a,si,0)  # like a[si,:] but general for ndarray's

        # Return similar datastructure to what we got.  Convert all to numpy
        out = [d.asnumpy() for d in data]
        out[self.shuffle_data_idx] = shuf_a
        return out

    def _refill_queue(self):
        """Fetch another batch from the source, and shuffle it to make
        negative samples.
        """
        original = self._sourcedata.next().data  # List of NDArrays: one per input
        batch_size = original[0].shape[0]
        num_inputs = len(original)

        #Start with positive examples, copied straight
        outdata = [[o.asnumpy()] for o in original] # list of lists of numpy arrays
        outlabels = [np.ones(batch_size) * self.positive_label] # list of numpy arrays
        # The inner list of both is the set of samples.  We'll recombine later.

        # Construct negative samples.
        for _ in range(self.sample_ratio):
            shuffled = self._shuffle_batch(original)
            for i,d in enumerate(shuffled):
                outdata[i].append(d)
            outlabels.append(np.ones(batch_size) * self.negative_label)
        def stacker(x):
            if len(x[0].shape)==1:
                return np.hstack(x)
            else:
                return np.vstack(x)
        outdata = [stacker(x) for x in outdata] # Single tall vectors
        outlabels = stacker(outlabels)

        # Record-level shuffle so the negatives are mixed in.
        def shuffler(x, idx):
            return np.take(x,idx,0)
        shuf_idx = np.arange(len(outlabels))
        np.random.shuffle(shuf_idx)
        outdata = [shuffler(o,shuf_idx) for o in outdata]
        outlabels = shuffler(outlabels,shuf_idx)
        self._push_queue(outdata,outlabels)


if __name__ == "__main__":
    print("Simple test of NegativeSamplingDataIter")
    np.random.seed(123)
    A = np.random.randint(-20,150,size=(100,5))
    B = np.random.randint(-2,15,size=(100,2))
    R = np.random.randint(1,5,size=(100,))
    batch_size=3
    oridi = mx.io.NDArrayIter(data={'a':A,'b':B},label=R, batch_size=batch_size)
    oribatch = oridi.next()
    oridi.reset()
    for ratio in range(0,5):
        nsdi = NegativeSamplingDataIter(oridi, sample_ratio=ratio)

        # Check sizes of output
        bat = nsdi.next()
        for i in range(len(bat.data)):
            assert bat.data[i].shape[0] == batch_size
            assert bat.data[i].shape[1] == oribatch.data[i].shape[1]
        assert bat.label.shape[0] == batch_size

        # Check that we get more minibatches
        oridi.reset()
        ori_cnt = len(list(oridi))
        nsdi.reset()
        ns_cnt = len(list(nsdi))
        assert ns_cnt == ori_cnt * (1+ratio)

    print("Tests done")

