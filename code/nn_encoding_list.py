from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import random

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten, MaxPooling2D, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras import regularizers

"""Layer types

	FC:1
	Conv:2
	Pooling:3
	RNN:4
	PerturbateParam:5
	Empty:6

"""

"""Layer tuple

	0: Layer type
	1: Neuron number
	2: Activation function (0:sigmoid, 1:tanh, 2:relu, 3:linear, 4:softmax)
	3: Filters for cnn
	4: Kernel_size for cnn (just one for squared kernels)
	5: Strides for cnn

"""

ann_building_rules = {
	
					1:[1],
					2:[1, 2, 3],
					3:[1, 2],
					4:[1, 4],
					5:[],
					6:[1, 2, 3, 4]
}

def generate_model(model=None, prev_component=6, next_component=6, max_layers=64, more_layers_prob=0.8):
	"""Iteratively and randomly generate a model"""

	layer_count = 0
	success = False

	if model == None:
		model = list()


	while True:

		curr_component = random.choice(ann_building_rules[prev_component])

		if curr_component == 5:
			ann_building_rules[5] = ann_building_rules[prev_component]

		rndm = random.random()
		#print(rndm)
		more_layers = (rndm <= more_layers_prob)
		#print(more_layers)

		#Keep adding more layers
		if more_layers == False:
			#Is this layer good for ending?
			if next_component == 6 or next_component in ann_building_rules[curr_component]:
				layer = generate_layer(curr_component)
				model.append(layer)
				prev_component = curr_component
				success = True
				break

			#Keep looking for layer	if max_layers not reached
			elif max_layers >= layer_count:
				continue
			else:
				success = False
				model = []
				break
		else:
			layer = generate_layer(curr_component)
			model.append(layer)
			prev_component = curr_component

	return model, success


def generate_layer(layer_type):
	"""Given a layer type, return the layer params"""

	layer = [layer_type, 0, 0, 0, 0, 0]
	layer[1] = 8*random.randint(1,129) #Generate a random number of neurons which is a multiple of 8 up to 1024 neurons
	layer[2] = random.randint(0,3) if layer_type != 5 else 2
	layer[3] = 8*random.randint(1,64)
	layer[4] = 3**random.randint(1,6)
	layer[5] = random.randint(1,6)

	return layer


	"""
	if layer_type == 2: #FC layer
	elif layer_type == 3: #Conv layer
	elif layer_type == 4: #Pooling layer
	elif layer_type == 5: #RNN layer
	else:
	"""

def decode_genotype(model_genotype):
	"""From a model genotype, generate the keras model"""

	model = Sequential()

	print(model_genotype)

	for i in range(len(model_genotype)):

		print(i)
		layer = model_genotype[i]
		print(layer)




"""Input can be of 3 types, ANN (1), CNN (2) or RNN (4)"""
def main():

	architecture_type = 4
	problem_type = 1  #1 for regression, 2 for classification
	number_classes = 8 #If regression applies, number of classes

	model, success = generate_model(more_layers_prob=0.9, prev_component=architecture_type)

	#Generate first layer
	layer_first = generate_layer(architecture_type)

	#Last layer is always FC
	if problem_type == 1:
		layer_last = [1, 1, 3, 0, 0, 0]
	else:
		layer_last = [1, number_classes, 4, 0, 0, 0]

	#print(model)

	model.append(layer_last)
	model_full = [layer_first] + model

	#print(model_full)
	decode_genotype(model_full)



main()
















