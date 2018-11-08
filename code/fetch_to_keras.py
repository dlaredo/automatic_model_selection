import random
import CMAPSAuxFunctions

import keras
import keras.layers
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras import regularizers

from ann_encoding_rules import Layers, ann_building_rules, activations

from tunable_model import SequenceTunableModelRegression


def create_tunable_model(model_genotype, problem_type, input_shape, data_handler, model_number):

	#K.clear_session()  #Clear the previous tensorflow graph
	model = decode_genotype(model_genotype, problem_type, input_shape, 1)

	"""
	if model != None:
		model.summary()
	"""

	lrate = LearningRateScheduler(CMAPSAuxFunctions.step_decay)

	model = get_compiled_model(model, problem_type, optimizer_params=[])
	tModel = SequenceTunableModelRegression('ModelMNIST_SN_'+str(model_number), model, lib_type='keras', data_handler=data_handler)

	return tModel


def population_to_keras(population, input_shape, data_handler):

	for i in range(len(population)):

		individual = population[i]

		tModel = create_tunable_model(individual.stringModel, individual.problem_type, input_shape, data_handler, i+1)

		#Create the individual
		individual.tModel = tModel
		#individual = Individual(i, problem_type, model_genotype)


def decode_genotype(model_genotype, problem_type, input_shape, output_dim):
	"""From a model genotype, generate the keras model"""

	model = Sequential()
	return_sequences = False

	#Fetch first layer
	curr_layer = model_genotype[0]
	curr_layer_type = curr_layer[0]
	next_layer_type = model_genotype[1][0]

	if next_layer_type == Layers.Recurrent:  #If the next layer is LSTM return sequences is true
		return_sequences = True

	klayer = array_to_layer(curr_layer, problem_type=problem_type, input_shape=input_shape, first_layer=True, return_sequences=return_sequences)

	if klayer != None:
		model.add(klayer)
	else:
		print("Model could not be fetched")
		return None

	if next_layer_type == Layers.FullyConnected:
		if curr_layer_type == Layers.Convolutional: #If the next layer is FC and current is Conv, Flatten output
			model.add(keras.layers.Flatten())

	#Fetch layers 1,...,l
	for i in range(1,len(model_genotype)):

		return_sequences = False

		curr_layer = model_genotype[i]
		curr_layer_type = curr_layer[0]
		next_layer_type = model_genotype[i][0]

		if next_layer_type == Layers.Recurrent:  #If the next layer is LSTM return sequences is true
			return_sequences = True

		klayer = array_to_layer(curr_layer, return_sequences=return_sequences)

		if klayer != None:
			model.add(klayer)
		else:
			print("Model could not be fetched")
			return None

		if next_layer_type == Layers.FullyConnected:
			if curr_layer_type == Layers.Convolutional: #If the next layer is FC and current is Conv, Flatten output
				model.add(keras.layers.Flatten())

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

		if array[0] == Layers.FullyConnected:
			klayer = keras.layers.Dense(neurons_units, input_shape=input_shape, activation=activations[activation], kernel_initializer='glorot_normal', name='in')
		elif array[0] == Layers.Recurrent:
			klayer = keras.layers.LSTM(neurons_units, input_shape=input_shape, activation=activations[activation], kernel_initializer='glorot_normal', 
				return_sequences=return_sequences, name='in')
		elif array[0] == Layers.Convolutional:
			klayer = keras.layers.Conv2D(filter_size, (kernel_size, kernel_size), strides=(stride, stride), input_shape=input_shape, padding='valid',
				activation=activations[activation], kernel_initializer='glorot_normal', name='in')
		else:
			print("Layer not valid for the first layer")
			klayer = None
	else:

		if array[0] == Layers.FullyConnected:
			klayer = keras.layers.Dense(neurons_units, activation=activations[activation], kernel_initializer='glorot_normal')
		elif array[0] == Layers.Recurrent:
			klayer = keras.layers.LSTM(neurons_units, activation=activations[activation], kernel_initializer='glorot_normal', return_sequences=return_sequences)
		elif array[0] == Layers.Convolutional:
			klayer = keras.layers.Conv2D(filter_size, (kernel_size, kernel_size), strides=(stride, stride), padding='valid',
				activation=activations[activation], kernel_initializer='glorot_normal')
		elif array[0] == Layers.Pooling:
			klayer = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))
		elif array[0] == Layers.Dropout:
			klayer = keras.layers.Dropout(dropout_rate)
		else:
			print("Layer not valid")
			klayer = None

	return klayer


def get_compiled_model(model, problem_type, optimizer_params=[]):
	"""Obtain a keras compiled model"""
	
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





