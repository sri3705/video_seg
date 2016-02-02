#In the name of God
import caffe
import numpy as np


class MyHingLossDotProductLayer(caffe.Layer):
    """
    Compute the hing loss of the dot product of the 2 input layers
    """
# max(0, 1 - a.b)
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute dot product.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        top[0].reshape(1)

    def forward(self, bottom, top):
        hinge = np.vectorize(lambda x: max(0, x), otypes=[bottom[0].data.dtype])
        self.res = hinge(np.ones(bottom[0].data.shape[0]) - np.sum(bottom[0].data * bottom[1].data, axis=1))
        top[0].data[...] = np.sum(self.res) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        hing_res = np.sign(self.res)
        for i in range(2):
            if not propagate_down[i]:
                continue
        for d in range(bottom[0].data.shape[0]):
            bottom[i].diff[d][...] = -1 * hing_res[d] * bottom[1-i].data[d][...]

        bottom[i].diff[...] /= bottom[i].num
