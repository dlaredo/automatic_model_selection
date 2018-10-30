import random

import keras
import keras.layers
from keras.models import Sequential, Model
#from keras.layers import Dense, Input, Dropout, Reshape, Conv2D, Flatten, MaxPooling2D, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras import regularizers

from data_handler_MNIST import MNISTDataHandler

from tunable_model import SequenceTunableModelRegression

import CMAPSAuxFunctions
from CMAPSAuxFunctions import TrainValTensorBoard

"""Layer types

	FC:1
	Conv:2
	RNN:3
	Pooling:4
	Dropout:5
	PerturbateParam:6
	Empty:7

"""

"""Layer tuple

	0: Layer type
	1: Neuron number
	2: Activation function (0:sigmoid, 1:tanh, 2:relu, 3:linear, 4:softmax)
	3: Filters for cnn
	4: Kernel_size for cnn (just one for squared kernels)
	5: Strides for cnn
	6: Pooling size
	7: Dropout rate

"""

ann_building_rules = {
	
					1:[1, 5],
					2:[1, 2, 4, 5],
					3:[1, 3, 5],
					4:[1, 2],
					5:[1, 2, 3],
					6:[],
					7:[1, 2, 3, 4]
}

activations = {0:'sigmoid', 1:'tanh', 2:'relu', 3:'softmax', 4:'linear'}


def generate_model(model=None, prev_component=7, next_component=7, max_layers=64, more_layers_prob=0.8):
	"""Iteratively and randomly generate a model"""

	layer_count = 0
	success = False

	if model == None:
		model = list()

	while True:

		curr_component = random.choice(ann_building_rules[prev_component])

		#Perturbate param layer
		if curr_component == 6:
			ann_building_rules[6] = ann_building_rules[prev_component]
		elif curr_component == 5: #Dropout layer
			ann_building_rules[5] = ann_building_rules[prev_component].copy()
			ann_building_rules[5].remove(5)

		rndm = random.random()
		#print(rndm)
		more_layers = (rndm <= more_layers_prob)
		#print(more_layers)

		#Keep adding more layers
		if more_layers == False:
			#Is this layer good for ending?
			if next_component == 7 or next_component in ann_building_rules[curr_component]:
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

	layer = [layer_type, 0, 0, 0, 0, 0, 0, 0]
	layer[1] = 8*random.randint(1,129) #Generate a random number of neurons which is a multiple of 8 up to 1024 neurons
	layer[2] = random.randint(0,3) if layer_type != 5 else 2
	layer[3] = 8*random.randint(1,64)
	layer[4] = 3**random.randint(1,6)
	layer[5] = random.randint(1,6)
	layer[6] = 2**random.randint(1,6)
	layer[7] = random.uniform(0, 0.5)

	return layer


def decode_genotype(model_genotype, problem_type, input_shape, output_dim):
	"""From a model genotype, generate the keras model"""

	model = Sequential()
	return_sequences = False

	#print(model_genotype)

	#Fetch first layer
	curr_layer = model_genotype[0]
	curr_layer_type = curr_layer[0]
	next_layer_type = model_genotype[1][0]

	#print("\n\n\n")
	#print(curr_layer)

	if next_layer_type == 3:  #If the next layer is LSTM return sequences is true
		return_sequences = True

	klayer = array_to_layer(curr_layer, problem_type=problem_type, input_shape=input_shape, first_layer=True, return_sequences=return_sequences)
	#print(klayer)

	if klayer != None:
		model.add(klayer)
	else:
		print("Model could not be fetched")
		return None

	if next_layer_type == 1:
		if curr_layer_type == 2: #If the next layer is FC and current is Conv, Flatten output
			model.add(keras.layers.Flatten())

	#Fetch layers 1,...,l-1
	for i in range(1,len(model_genotype)):

		return_sequences = False

		curr_layer = model_genotype[i]
		curr_layer_type = curr_layer[0]
		next_layer_type = model_genotype[i][0]

		#print(curr_layer)

		if next_layer_type == 3:  #If the next layer is LSTM return sequences is true
			return_sequences = True

		klayer = array_to_layer(curr_layer, return_sequences=return_sequences)
		#print(klayer)

		if klayer != None:
			model.add(klayer)
		else:
			print("Model could not be fetched")
			return None

		if next_layer_type == 1:
			if curr_layer_type == 2: #If the next layer is FC and current is Conv, Flatten output
				model.add(keras.layers.Flatten())

	#Fetch last layer
	"""
	curr_layer = model_genotype[-1]
	print(curr_layer)
	klayer = array_to_layer(curr_layer, problem_type=problem_type, output_dim=output_dim, last_layer=True)
	#print(klayer)

	if klayer != None:
		model.add(klayer)
	else:
		print("Model could not be fetched")
		return None

	model.add(klayer)"""

	return model



def array_to_layer(array, problem_type=0, input_shape=(0,), output_dim=0, first_layer=False, last_layer=False, return_sequences=False):
	"""Map from an array to a layer"""

	klayer = None
	neurons_units = array[1]
	activation = array[2]
	filter_size = array[3]
	kernel_size = array[4]
	stride = array[5]
	pool_size = array[6]
	dropout_rate = array[7]

	if first_layer == True:

		if array[0] == 1:
			klayer = keras.layers.Dense(neurons_units, input_shape=input_shape, activation=activations[activation], kernel_initializer='glorot_normal', name='in')
		elif array[0] == 2:
			klayer = keras.layers.LSTM(neurons_units, input_shape=input_shape, activation=activations[activation], kernel_initializer='glorot_normal', 
				return_sequences=return_sequences, name='in')
		elif array[0] == 3:
			klayer = keras.layers.Conv2D(filter_size, (kernel_size, kernel_size), strides=(stride, stride), input_shape=input_shape, padding='valid',
				activation=activations[activation], kernel_initializer='glorot_normal', name='in')
		else:
			print("Layer not valid for the first layer")
			klayer = None
	else:

		if array[0] == 1:
			klayer = keras.layers.Dense(neurons_units, activation=activations[activation], kernel_initializer='glorot_normal')
		elif array[0] == 2:
			klayer = keras.layers.LSTM(neurons_units, activation=activations[activation], kernel_initializer='glorot_normal', return_sequences=return_sequences)
		elif array[0] == 3:
			klayer = keras.layers.Conv2D(filter_size, (kernel_size, kernel_size), strides=(stride, stride), padding='valid',
				activation=activations[activation], kernel_initializer='glorot_normal')
		elif array[0] == 4:
			klayer = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))
		elif array[0] == 5:
			klayer = keras.layers.Dropout(dropout_rate)
		else:
			print("Layer not valid")
			klayer = None

	return klayer


def get_compiled_model(model, problem_type, optimizer_params=[]):
	"""Obtain a keras compiled model"""
	
	#To test the model without randomness

	
	#Shared parameters for the models
	optimizer = Adam(lr=0, beta_1=0.5)
	
	if problem_type == 1:
		lossFunction = "mean_squared_error"
		metrics = ["mse"]
	elif problem_type == 2:
		lossFunction = "categorical_crossentropy"
		metrics = ["accuracy"]
	else:
		print("Problem type not defined")
		model = None
		return 	

	#Create and compile the models
	model.compile(optimizer = optimizer, loss = lossFunction, metrics = metrics)

	return model


def partial_run(model, problem_type, data_handler, cross_validation_ratio, run_number):
	"""This should be run in Ray"""

	lrate = LearningRateScheduler(CMAPSAuxFunctions.step_decay)

	"""How to keep the data in a way such that it doesnt create too much overhead"""
	model = get_compiled_model(model, problem_type, optimizer_params=[])
	tModel = SequenceTunableModelRegression('ModelMNIST_SN_'+str(run_number), model, lib_type='keras', data_handler=data_handler)
	tModel.load_data(verbose=1, cross_validation_ratio=0.2)
	tModel.print_data()

	tModel.epochs = 20
	tModel.train_model(learningRate_scheduler=lrate, verbose=1)

	tModel.evaluate_model(cross_validation=True)

	cScores = tModel.scores
	print(cScores)

	tModel.evaluate_model(cross_validation=False)

	cScores = tModel.scores
	print(cScores)



def main():
	"""Input can be of 3 types, ANN (1), CNN (2) or RNN (3)"""

	architecture_type = 1
	problem_type = 2  #1 for regression, 2 for classification
	number_classes = 10 #If regression applies, number of classes
	input_shape = (784,)
	cross_val = 0.2


	K.clear_session()  #Clear the previous tensorflow graph

	model, success = generate_model(more_layers_prob=0.5, prev_component=architecture_type)

	#Generate first layer
	layer_first = generate_layer(architecture_type)

	#Last layer is always FC
	if problem_type == 1:
		layer_last = [1, 1, 4, 0, 0, 0, 0, 0]
	else:
		layer_last = [1, number_classes, 3, 0, 0, 0, 0, 0]

	#print(model)

	model.append(layer_last)
	model_full = [layer_first] + model

	print(model_full)
	model = decode_genotype(model_full, problem_type, input_shape, 1)

	if model != None:
		model.summary()


	#Test using mnist
	dHandler_mnist = MNISTDataHandler()
	partial_run(model, problem_type, dHandler_mnist, cross_val, 1)




main()
















