#In the name of God
import numpy as np
import matplotlib.pyplot as plt

import caffe

def print_layer_sizes(net):
	items = [(k, v.data.shape) for k, v in net.blobs.items()]
	print "Layers: "
	for i in items:
		print i 
	print 

def print_data(net, start=0):
	for i, key in enumerate(net.blobs):
		if i > start:
			print str(i)+ "-"+str(key)+": ", net.blobs[key].data
#	print "Context_sum:", net.blobs['context_sum'].data
#	print "target negative diff:", net.blobs['target_negative_diff'].data
#	print "embedding function context:", net.blobs['embedding_function_context'].data
#	print "embedding function target:", net.blobs['embedding_function_target'].data	
#	print "loss:", net.blobs['loss'].data

root = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/model/'

#caffe.set_decive(0)
caffe.set_mode_gpu()
net = caffe.Net(root + 'model.prototxt', caffe.TRAIN)

# TODO: Feed dummy data to the net and test if it works. So something like:
#net.blobs['data'].reshape(1,6,1,2)

#net.blobs['data'].data[...] = np.array([[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[5, 5]]],])

solver = caffe.SGDSolver(root+'solver.prototxt')

#solver.net.blobs['data'].data[...] = np.array([[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[0, 0]]],
#					[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[3, 2]]],
#					[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[-2, 3]]],
#					[[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],[[2, -2]]],
#					])

#print net.params['embedding_function_context'][0].data
#print net.params['embedding_function_target'][0].data
#solver.net.params['embedding_function_context'][0].data[...] = np.array([[1,0],[0,1]], dtype=np.float32)


#out = net.forward()


solver.net.forward()
solver.net.backward()


print_layer_sizes(solver.net)
print_data(solver.net, 8)
print solver.net.blobs['target'].data
#print solver.net.params['target'][0].data
#print solver.net.params['neighbor0'][0].data
#print solver.net.params['neighbor1'][0].data
print solver.net.params['inner_product_negative'][0].data
for i in range(5):
	solver.step(1)

#print_data(solver.net, 8)
print solver.net.params['inner_product_negative'][0].data
print solver.net.params['inner_product_target'][0].data
#print net.params['embedding_function_target'][0].data
#out1 = net.backward()
#print "Context_sum:", net.blobs['context_sum'].data
#print "target negative diff:", net.blobs['target_negative_diff'].data
#print "embedding function context:", net.blobs['embedding_function_context'].data
#print "embedding function target:", net.blobs['embedding_function_target'].data
#print net.params['embedding_function_context'].data
#print net.params['embedding_function_target'].data
#print net.params['embedding_function'].data
#print out
