import caffe
from numpy import zeros
import numpy as np

from configs import getConfigs

conf = getConfigs(8)


model_prototxt_path = conf.model['model_prototxt_path']
solver_prototxt_path = conf.solver['_solver_prototxt_path']


#root = '/cs/vml3/mkhodaba/cvpr16/code/embedding_segmentation/model/'

#caffe.set_decive(2)
caffe.set_mode_gpu()
#caffe.set_device(2)



test_interval = conf.solver['test_interval'] #10000
niter = conf.solver['max_iter'] #500000
train_interval = conf.solver['_train_interval'] #1000
termination_threshold = conf.solver['_termination_threshold']
net = caffe.Net(model_prototxt_path, caffe.TRAIN)

#solver = caffe.SGDSolver(root+'solver.prototxt')
solver = caffe.SGDSolver(solver_prototxt_path)
# losses will also be stored in the log
#train_loss = zeros(niter)
train_loss = np.array([])
test_acc = zeros(int(np.ceil(niter / test_interval)))
#output = zeros((niter, 8, 10))
test_loss = 0
# the main solver loop
it = -1
prev_loss = 100000000
diff_loss = 100

while it < niter and abs(diff_loss) >= termination_threshold:
	it += 1
#for it in xrange(niter):
	#print 'iter', it
	solver.step(1)  # SGD by Caffe
	#print 'salam'
    	#print 'step done'
    # store the train loss
	#target_data = solver.net.blobs['target'].data
	
	#print 'target_data', target_data
	#print target_data.shape
	train_loss= np.append(train_loss, solver.net.blobs['loss'].data)
#	if it > 0 and train_loss[it-1] == 0:
#		train_loss[it-1] = train_loss[it]
	if it % train_interval == 0:
		print 'Iteration', it, '...'
		#print 'Loss:', train_loss[it]
		current_loss = np.mean(train_loss[-1000:])
		prev_loss, diff_loss = current_loss, prev_loss-current_loss
		print 'Average Train Loss [last 1000]:', current_loss, '-- Train Loss Std', np.std(train_loss[-1000:])
		print 'Minimum Train Loss [last 1000]:', np.amin(train_loss[-1000:])	
		print 'Improvement [last 1000]:', diff_loss

		#print 'Test Loss so far:', test_loss
	#solver.test_nets[0].forward(start='conv1')
	#TODO TESTING!!!!
	
	#if it % test_interval == 0:
	#	for test_it in xrange(100): #len supervoxels
	#		solver.test_nets[0].forward() 
			#TODO now get the representations
		#TODO compute distance
		#TODO mAP
	#	print 'Test Loss:', solver.test_nets[0].blobs['loss'].data
	#	test_loss = solver.test_nets[0].blobs['loss'].data

#np.savetxt(root+'embedding.txt', solver.net.params['inner_product_target'][0].data)
#with open(root+'embedding.txt', 'w') as f:
#	f.write(str(solver.net.params['inner_product_target'][0].data))
#print solver.net.params['inner_product_negative'][0].data

#print solver.net.params['inner_product_target'][0].data


    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    #solver.test_nets[0].forward(start='conv1')
    #output[it] = solver.test_nets[0].blobs['ip2'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    #if it % test_interval == 0:
    #   print 'Iteration', it, 'testing...'
    #    correct = 0
    #    for test_it in range(100):
    #        solver.test_nets[0].forward()
    #        correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
    #                       == solver.test_nets[0].blobs['label'].data)
    #    test_acc[it // test_interval] = correct / 1e4
