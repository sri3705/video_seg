#In the name of God

from caffe import layers as L
from caffe import params as P
from caffe import NetSpec

class Network:
	def __init__(self, number_of_neighbors, num_output=100, lr_mult):
		self.number_of_neighbors = number_of_neighbors
		#self.netP = 
		self.net = NetSpec()
		self.shared_weight_counter = 0
		self.num_output = num_output

	def getInnerProduct(self, input_name, output_name, ID, num_output=None):
		#TODO What should be the output of this layer?
		if num_output is None:
			num_output = self.num_output
		return L.InnerProduct(getattr(self.net, input_name),
						name=output_name,
						weight_filler=dict(type='xavier'),
						bias_filler=dict(type='xavier', value=0.2),
						num_output=num_output,
						#in_place=True,
						param=list([dict(name="embed_w{0}".format(ID), lr_mult=1, decay_mult=1), 
							    dict(name="embed_b{0}".format(ID), lr_mult=2, decay_mult=0)])
						)

	def createEmbeddingNetwork(self, dataset_path, batch_size):
		dataLayer = L.HDF5Data(name='dataLayer', 
						source=dataset_path, 
						batch_size=batch_size, 
						ntop=2+self.number_of_neighbors)# tops-> target, [neighbors], negative
		#data -> [target, neighbor1, neighbor2, ..., neighbork, negative]
		self.net.target = dataLayer[0]
		self.net.negative = dataLayer[-1]
		for l in range(1, self.number_of_neighbors+1):
			setattr(self.net, 'neighbor{0}'.format(l-1), dataLayer[l])		

		
		#First layer of inner product 
		self.net.inner_product_target = self.getInnerProduct('target', 'inner_product_target', 1)
		self.net.inner_product_negative = self.getInnerProduct('negative', 'inner_product_negative', 1)
		for i in range(0, self.number_of_neighbors):
			layer = self.getInnerProduct('neighbor{0}'.format(i), 'inner_product_neighbor{0}'.format(i), 1)
			setattr(self.net, 'inner_product_neighbor{0}'.format(i), layer)
		
		#ReLU
		self.net.relu_target = L.ReLU(self.net.inner_product_target, name='relu_target', in_place=True)
		self.net.relu_negative = L.ReLU(self.net.inner_product_negative, name='relu_negative', in_place=True)
		for i in range(0, self.number_of_neighbors):
			layer = L.ReLU(getattr(self.net, 'inner_product_neighbor{0}'.format(i)), 
					name='relu_neighbor{0}'.format(i),
					in_place=True)
			setattr(self.net, 'relu_neighbor{0}'.format(i), layer)
		
		#Second layer of inner product
		#self.net.inner_product2_target = self.getInnerProduct('inner_product_target', 'inner_product2_target', 2)
		#self.net.inner_product2_negative = self.getInnerProduct('inner_product_negative', 'inner_product2_negative', 2)
		#for i in range(0, self.number_of_neighbors):
		#	layer = self.getInnerProduct('inner_product_neighbor{0}'.format(i), 
		#					'inner_product2_neighbor{0}'.format(i), 2)
		#	setattr(self.net, 'inner_product2_neighbor{0}'.format(i), layer)
			
		#Context
		'''
		context_sum_bottom = []
		for i in range(0, self.number_of_neighbors):
			context_sum_bottom.append(getattr(self.net, 'inner_product2_neighbor{0}'.format(i)))
		coeff = 1.0/self.number_of_neighbors		
		self.net.context_sum = L.Eltwise(*context_sum_bottom,
						name='context_sum',
						operation=P.Eltwise.SUM, # 1 -> SUM
						coeff=list([coeff for i in range(self.number_of_neighbors)]))
		
		#Target - Negative
		self.net.target_negative_diff = L.Eltwise(self.net.inner_product2_target, self.net.inner_product2_negative,
								name='target_negative_diff',
								operation=P.Eltwise.SUM, # SUM
								coeff=list([1,-1])) # target - negative
		'''
		#Context
		context_sum_bottom = []
		for i in range(0, self.number_of_neighbors):
			context_sum_bottom.append(getattr(self.net, 'inner_product_neighbor{0}'.format(i)))
		coeff = 1.0/self.number_of_neighbors		
		self.net.context_sum = L.Eltwise(*context_sum_bottom,
						name='context_sum',
						operation=P.Eltwise.SUM, #  SUM
						coeff=list([coeff for i in range(self.number_of_neighbors)]))
		
		#Target - Negative
		self.net.target_negative_diff = L.Eltwise(self.net.inner_product_target, self.net.inner_product_negative,
								name='target_negative_diff',
								operation=P.Eltwise.SUM, # SUM
								coeff=list([1,-1])) # target - negative
		

		#Loss layer
		self.net.loss = L.Python(self.net.context_sum, self.net.target_negative_diff,
						name='loss',
						module='my_dot_product_layer',
						layer='MyHingLossDotProductLayer')

	def saveNetwork(self, path):
		with open(path, 'w') as f:
			f.write("force_backward:true\n"+str(self.net.to_proto()))

def createNetwork(settings):
	dataset_path = 		settings['dataset_list_path']
	neighbor_num = 		settings['neighbor_num']
	inner_product_output = 	settings['inner_product_output']
	batch_size = 		settings['batch_size']
	network = Network(neighbor_num, inner_product_output)
	network.createEmbeddingNetwork(dataset_path, batch_size)
	network.saveNetwork(model_path)

