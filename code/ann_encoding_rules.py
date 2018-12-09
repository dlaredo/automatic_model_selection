import random

from enum import Enum

max_neuron_multiplier = 1024/8
max_filter_size_multiplier = 512/8
max_filter_size_exponent = 6
max_filter_stride = 6
max_pooling_exponent = 6
max_dropout = 0.7


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


def generate_layer(layer_type, used_activations={}):
	"""Given a layer type, return the layer params

	0: Type of layer
	1: Number of neurons for fully connected layers
	2: Type of activation function
	3: CNN filter size
	4: CNN kernel size
	5: CNN stride
	6: Pooling size
	7: Dropout rate
	"""

	layer = [layer_type, 0, 0, 0, 0, 0, 0, 0]

	if layer_type == Layers.FullyConnected or layer_type == Layers.Recurrent:
		layer[1] = 8*random.randint(1, max_neuron_multiplier) #Generate a random number of neurons which is a multiple of 8 up to 1024 neurons

	#Use the same activation for all the similar layers of the network
	if layer_type == Layers.FullyConnected or layer_type == Layers.Convolutional or layer_type == Layers.Recurrent:
		if layer_type in used_activations:
			layer[2] = used_activations[layer_type]
		else:
			layer[2] = random.randint(0,2) if layer_type != Layers.Recurrent else 1  #Exclude softmax since that only goes till the end
			used_activations[layer_type] = layer[2]


	if layer_type == Layers.Convolutional:
		layer[3] = 8*random.randint(1, max_filter_size_multiplier)
		layer[4] = 3**random.randint(1, max_filter_size_exponent)
		layer[5] = random.randint(1, max_filter_stride)

	if layer_type == Layers.Pooling:
		layer[6] = 2**random.randint(1, max_pooling_exponent)

	if layer_type == Layers.Dropout:
		#Dropout is of the form 0.1, 0.15, 0.2, 0.25, ....
		dropout_ratio = random.uniform(0.1, max_dropout)
		dropout_ratio = round(rectify_dropout_ratio(dropout_ratio),2)

		layer[7] = dropout_ratio

	return layer






