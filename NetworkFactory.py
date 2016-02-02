from caffe import layers as L
from caffe import params as P
from caffe import NetSpec
from utils import getFuncArgNames

class Network:
    def __init__(self, number_of_neighbors=6, inner_product_output=100, weight_lr_mult=1, weight_decay_mult=1, b_lr_mult=2, b_decay_mult=0):
        self.number_of_neighbors = number_of_neighbors
        #self.netP =
        self.net = NetSpec()
        self.shared_weight_counter = 0
        self.num_output = inner_product_output
        self.weight_lr_mult = weight_lr_mult
        self.weight_decay_mult = weight_decay_mult
        self.b_lr_mult = b_lr_mult
        self.b_decay_mult = b_decay_mult

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
            			param=list([dict(name="embed_w{0}".format(ID), lr_mult=self.weight_lr_mult, decay_mult=self.weight_decay_mult),
            				    dict(name="embed_b{0}".format(ID), lr_mult=self.b_lr_mult, decay_mult=self.b_decay_mult)])
            			)


    def l2normed(self,vec, dim):
        #Returns L2-normalized instances of vec; i.e., for each instance x in vec,
        #computes  x / ((x ** 2).sum() ** 0.5). Assumes vec has shape N x dim."""
        denom = L.Reduction(vec, axis=1, operation=P.Reduction.SUMSQ)
        denom = L.Power(denom, power=(-0.5), shift=1e-12)
        denom = L.Reshape(denom, num_axes=0, axis=-1, shape=dict(dim=[1]))
        denom = L.Tile(denom, axis=1, tiles=dim)

        return L.Eltwise(vec, denom, operation=P.Eltwise.PROD)

    def createEmbeddingNetwork(self, database_list_path='.', batch_size=20, phase=0):
        dataset_path = database_list_path
        dataLayer = L.HDF5Data(name='dataLayer',
            			source=dataset_path,
            			batch_size=batch_size,
            			ntop=2+self.number_of_neighbors,
            			include=list([dict(phase=phase)]))# tops-> target, [neighbors], negative
        #data -> [target, neighbor1, neighbor2, ..., neighbork, negative]
        self.net.target = dataLayer[0]
        self.net.negative = dataLayer[-1]
        for l in range(1, self.number_of_neighbors+1):
            setattr(self.net, 'neighbor{0}'.format(l-1), dataLayer[l])

        self.net.target_norm1 = L.MVN(self.net.target, name='norm_target')
        self.net.negative_norm1 = L.MVN(self.net.negative, name='norm_negative')
        for l in range(1, self.number_of_neighbors+1):
            layer = L.MVN(getattr(self.net, 'neighbor{0}'.format(l-1)), name='norm_neighbor{0}'.format(l-1))
            setattr(self.net, 'neighbor{0}_norm'.format(l-1), layer)

        #First layer of inner product
        self.net.inner_product_target = self.getInnerProduct('target_norm1', 'inner_product_target', 1)
        self.net.inner_product_negative = self.getInnerProduct('negative_norm1', 'inner_product_negative', 1)
        for i in range(0, self.number_of_neighbors):
            layer = self.getInnerProduct('neighbor{0}_norm'.format(i), 'inner_product_neighbor{0}'.format(i), 1)
            setattr(self.net, 'inner_product_neighbor{0}'.format(i), layer)

        #First layer of inner product
        #self.net.inner_product_target = self.getInnerProduct('target', 'inner_product_target', 1)
        #self.net.inner_product_negative = self.getInnerProduct('negative', 'inner_product_negative', 1)
        #for i in range(0, self.number_of_neighbors):
        #    layer = self.getInnerProduct('neighbor{0}'.format(i), 'inner_product_neighbor{0}'.format(i), 1)
        #    setattr(self.net, 'inner_product_neighbor{0}'.format(i), layer)

        #ReLU
        self.net.relu_target = L.ReLU(self.net.inner_product_target, name='relu_target', in_place=True)
        self.net.relu_negative = L.ReLU(self.net.inner_product_negative, name='relu_negative', in_place=True)
        for i in range(0, self.number_of_neighbors):
            layer = L.ReLU(getattr(self.net, 'inner_product_neighbor{0}'.format(i)),
            		name='relu_neighbor{0}'.format(i),
            		in_place=True)
            setattr(self.net, 'relu_neighbor{0}'.format(i), layer)

        #self.net.normalize_target = L.NormalizeLayer(self.net.inner_product_target, name='normalize_target', in_place=True)
        #self.net.normalize_negative = L.NormalizeLayer(self.net.inner_product_negative, name='normalize_negative', in_place=True)
        #for i in range(0, self.number_of_neighbors):
            #layer = L.NormalizeLayer(getattr(self.net, 'inner_product_neighbor{0}'.format(i)),
            #		name='normalize_neighbor{0}'.format(i),
            		#in_place=True)
            #setattr(self.net, 'normalize_neighbor{0}'.format(i), layer)

        #Second layer of inner product
        #self.net.inner_product2_target = self.getInnerProduct('inner_product_target', 'inner_product2_target', 2)
        #self.net.inner_product2_negative = self.getInnerProduct('inner_product_negative', 'inner_product2_negative', 2)
        #for i in range(0, self.number_of_neighbors):
        #    layer = self.getInnerProduct('inner_product_neighbor{0}'.format(i),
        #    				'inner_product2_neighbor{0}'.format(i), 2)
        #    setattr(self.net, 'inner_product2_neighbor{0}'.format(i), layer)

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

        #Next section: normalizing embedding

        #Target - Negative
        #self.net.target_norm = self.l2normed(self.net.inner_product_target, self.num_output)
        #self.net.negative_norm = self.l2normed(self.net.inner_product_negative, self.num_output)
        #self.net.context_norm = self.l2normed(self.net.context_sum, self.num_output)

        #self.net.target_negative_diff = L.Eltwise(self.net.target_norm, self.net.negative_norm,
        #						name='target_negative_diff',
        #						operation=P.Eltwise.SUM, # SUM
        #						coeff=list([1,-1])) # target - negative

        #Loss layer
        #self.net.loss = L.Python(self.net.context_norm, self.net.target_negative_diff,
        #				name='loss',
        #				module='my_dot_product_layer',
        #				layer='MyHingLossDotProductLayer')

        #End of normalizing


        self.net.target_negative_diff = L.Eltwise(self.net.inner_product_target, self.net.inner_product_negative,
            					name='target_negative_diff',
            					operation=P.Eltwise.SUM, # SUM
            					coeff=list([1,-1])) # target - negative

        #Loss layer
        self.net.loss = L.Python(self.net.context_sum, self.net.target_negative_diff,
            			name='loss',
            			module='my_dot_product_layer',
            			layer='MyHingLossDotProductLayer')



    def saveNetwork(self, model_prototxt_path):
        with open(model_prototxt_path, 'w') as f:
            f.write("force_backward:true\n"+str(self.net.to_proto()))


def createNetwork(settings):
#    print settings
    network_input = {var_name:settings[var_name] for var_name in getFuncArgNames(Network.__init__) if var_name != 'self' and var_name in settings}
    network = Network(**network_input)

    inputs = {var_name:settings[var_name] for var_name in getFuncArgNames(Network.createEmbeddingNetwork) if var_name != 'self' and var_name in settings}
    network.createEmbeddingNetwork(**inputs)
#    print inputs

#    inputs = {var_name:db_settings[var_name] for var_name in getFuncArgNames(Network.createTestLayers) if var_name != 'self' and var_name in db_settings}
#    print inputs
#    network.createTestLayers(**inputs)

    inputs = {var_name:settings[var_name] for var_name in getFuncArgNames(Network.saveNetwork) if var_name != 'self' and var_name in settings}
    network.saveNetwork(**inputs)

def addTestLayers(configs):
    s = '''
    layer {{
      name: "dataLayer"
      type: "HDF5Data"
      top: "target"
      {0}
      top: "negative"
      include {{
        phase: TEST
      }}
      hdf5_data_param {{
        source: "{1}"
        batch_size: 1
    }}
      }}
    '''
    db = configs.db_settings
    action_name = db['action_name']
    database_path = db['test_database_list_path']
    neighbors = '\n'.join([ 'top:"neighbor{0}"'.format(i) for i in xrange(configs.model['number_of_neighbors'])])
    if db['db'] == 'jhmdb':
        level = db['level']
        video_name = db['video_name']
        with open(configs.model['model_prototxt_path'], 'r') as model_file:
            with open(configs.model['test_prototxt_path'], 'w') as test:
            	print configs.model['model_prototxt_path']
            	for action in action_name:
            		for i,video in enumerate(video_name[action]):
            			db_path = database_path.format(name=i)
            			print db_path
            			test.write(s.format(neighbors, db_path ))
            			#TODO Well breaking is not a good idea. It only writes
            			break
            	for line in model_file:
            		test.write(line)
    elif db['db'] == 'vsb100':
        with open(configs.model['model_prototxt_path'], 'r') as model_file:
            with open(configs.model['test_prototxt_path'], 'w') as test:
            	print configs.model['model_prototxt_path']
            	db_path = database_path.format(action_name=action_name)
            	test.write(s.format(neighbors, db_path))
            	for line in model_file:
            		test.write(line)
