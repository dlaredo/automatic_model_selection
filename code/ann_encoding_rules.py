import random

from enum import Enum

"""
layer_limits_global = {"max_neuron_multiplier":1024/8,
				"max_filter_size_multiplier":512/8,
				"max_kernel_size_multiplier":6,
				"max_filter_stride":3,
				"max_pooling_exponent":6,
				"max_dropout":0.7}
"""

"""
max_neuron_multiplier = 1024/8
max_filter_size_multiplier = 512/8
max_kernel_size_exponent = 6
max_filter_stride = 6
max_pooling_exponent = 6
max_dropout = 0.7
"""


class Layers(Enum):
	FullyConnected = 1
	Convolutional = 2
	Pooling = 3
	Recurrent = 4
	Dropout = 5
	PerturbateParam = 6
	Empty = 7


class LayerCharacteristics(Enum):
	LayerType = 0
	NeuronsNumber = 1
	ActivationType = 2
	FilterSizeCNN = 3
	KernelSizeCNN = 4
	StrideCNN = 5
	PoolingSize = 6
	DropoutRate = 7


ann_building_rules = {
	
					Layers.FullyConnected:[Layers.FullyConnected, Layers.Dropout],
					Layers.Convolutional:[Layers.FullyConnected, Layers.Convolutional, Layers.Pooling, Layers.Dropout],
					Layers.Pooling:[Layers.FullyConnected, Layers.Convolutional],
					Layers.Recurrent:[Layers.FullyConnected, Layers.Recurrent],
					Layers.Dropout:[Layers.FullyConnected, Layers.Convolutional, Layers.Recurrent],
					Layers.PerturbateParam:[],
					Layers.Empty:[Layers.FullyConnected, Layers.Convolutional, Layers.Recurrent]
}


activations = {0:'sigmoid', 1:'tanh', 2:'relu', 3:'softmax', 4:'linear'}


useful_components_by_layer = {
	
					Layers.FullyConnected:[1, 2],
					Layers.Convolutional:[2, 3, 4, 5],
					Layers.Pooling:[6],
					Layers.Recurrent:[1, 2],
					Layers.Dropout:[7]
}

activations_by_layer_type = {
	
					Layers.FullyConnected:[0, 1, 2],
					Layers.Convolutional:[1, 2],
					Layers.Recurrent:[1, 2],

}

def rectify_dropout_ratio(dropout_ratio):
	#Dropout is of the form 0.1, 0.15, 0.2, 0.25, ....

	dropout_ratio1 =(dropout_ratio*100)//10
	dropout_ratio2 = (dropout_ratio*100)%10

	dropout_ratio2 = 5 if dropout_ratio2 <= 5 else 0

	dropout_ratio = dropout_ratio1/10 + dropout_ratio2/100	

	return dropout_ratio


def generate_characteristic(layer, characteristic):
	"""Given a desired characteristic, generate a layer that effectively affects the layer"""

	value = -1
	layer_type = layer[0]

	if characteristic == LayerCharacteristics.NeuronsNumber:
		value = 8*random.randint(1, max_filter_size_multiplier) #Generate a random number of neurons which is a multiple of 8 up to 1024 neurons

	elif characteristic == LayerCharacteristics.ActivationType:
		value = random.randint(0,2) if layer_type != Layers.Recurrent else 1  #Exclude softmax since that only goes till the end

	elif characteristic == LayerCharacteristics.FilterSizeCNN:
		value = 8*random.randint(1, max_filter_size_multiplier)

	elif characteristic == LayerCharacteristics.KernelSizeCNN:
		value = 3**random.randint(1, max_filter_size_exponent)

	elif characteristic == LayerCharacteristics.StrideCNN:
		value = random.randint(1, max_filter_stride)

	elif characteristic == LayerCharacteristics.PoolingSize:
		value = 2**random.randint(1, max_pooling_exponent)

	elif characteristic == LayerCharacteristics.DropoutRate:
		
		value = random.uniform(0.1, max_dropout)
		value = round(rectify_dropout_ratio(value),2)

	else:
		pass

	return value


def generate_layer(layer_type, layer_limits, previous_layer=None, used_activations={}):
	"""Given a layer type, return the layer params

	0: Type of layer
	1: Number of neurons for fully connected layers
	2: Type of activation function
	3: CNN filter size
	4: CNN kernel size
	5: CNN stride
	6: Pooling size
	7: Dropout rate
	8: Layer output size
	"""

	print("Previous layer")
	print(previous_layer)

	"""Set the upper limits for the layers"""
	max_neuron_multiplier = layer_limits["max_neuron_multiplier"]
	max_pooling_exponent = layer_limits["max_pooling_exponent"]
	max_dropout = layer_limits["max_dropout"]

	layer = [layer_type, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	#If layer is convolutional try to generate it first, if its not possible then replace by fully connected
	if layer_type == Layers.Convolutional:
		convLayer = generate_convolutional_layer(previous_layer, layer_limits)

		if convLayer != None:
			print("Conv layer generated")
			layer[3] = convLayer[0]
			layer[4] = convLayer[1]
			layer[5] = convLayer[2]
		else:
			print("Impossible to generate conv layer, generating fully connected instead")
			layer_type = Layers.FullyConnected
			layer[0] = layer_type


	if layer_type == Layers.FullyConnected or layer_type == Layers.Recurrent:
		layer[1] = 8*random.randint(1, max_neuron_multiplier) #Generate a random number of neurons which is a multiple of 8 up to 1024 neurons

	#Use the same activation for all the similar layers of the network
	if layer_type == Layers.FullyConnected or layer_type == Layers.Convolutional or layer_type == Layers.Recurrent:
		if layer_type in used_activations:
			layer[2] = used_activations[layer_type]
		else:
			layer[2] = random.randint(0,2) if layer_type != Layers.Recurrent else 1  #Exclude softmax since that only goes till the end
			used_activations[layer_type] = layer[2]

	"""
	if layer_type == Layers.Convolutional:
		layer[3] = 8*random.randint(1, max_filter_size_multiplier)

		#Next layer uses smaller filter size
		layer[4] = 3**random.randint(1, max_kernel_size_multiplier)
		layer[5] = random.randint(1, max_filter_stride)
	"""

	if layer_type == Layers.Pooling:
		layer[6] = 2**random.randint(1, max_pooling_exponent)

	if layer_type == Layers.Dropout:
		#Dropout is of the form 0.1, 0.15, 0.2, 0.25, ....
		dropout_ratio = random.uniform(0.1, max_dropout)
		dropout_ratio = round(rectify_dropout_ratio(dropout_ratio),2)

		layer[7] = dropout_ratio

	return layer


def generate_convolutional_layer(previous_layer, layer_limits):
	""""""

	convLayer = None

	print("Layer limits")
	print(layer_limits)

	print("Previous Layer")
	print(previous_layer)

	print("generating convolutional layer")

	max_filter_size_multiplier = layer_limits["max_filter_size_multiplier"]
	max_kernel_size_multiplier = layer_limits["max_kernel_size_multiplier"]
	max_filter_stride = layer_limits["max_filter_stride"]

	filter_size = 8*random.randint(1, max_filter_size_multiplier)
	print("filter size %d"%filter_size)

	if previous_layer != None:
		previous_size = previous_layer[8]
	else:
		previous_size = max_kernel_size_multiplier

	print("previous size %d" % previous_size)
	max_kernel_size = previous_size // 2
	#max_kernel_size_multiplier = max_kernel_size // 3
	max_kernel_size_multiplier = max_kernel_size
	print("max kernel size %d " % max_kernel_size)
	print("max kernel size multiplier %d " % max_kernel_size_multiplier)

	# Can insert another conv layer
	if max_kernel_size_multiplier != 1:

		kernel_size = random.randint(1, max_kernel_size_multiplier) + 2 * random.randint(0, 1)

		if kernel_size / 2 == 0:
			kernel_size - 1

		stride = random.randint(1, max_filter_stride)

		convLayer = [filter_size, kernel_size, stride]

	return convLayer

