import random
import math

import numpy as np

from keras import backend as K

from ann_encoding_rules import Layers, ann_building_rules, activations
import fetch_to_keras
import CMAPSAuxFunctions

cross_validation_ratio = 0.2
epochs = 1
lrate = fetch_to_keras.LearningRateScheduler(CMAPSAuxFunctions.step_decay)

class Individual():


	def __init__(self, ind_label, problem_type, stringModel, tModel = None, raw_score = 0, raw_size = 0, fitness=0):
		self._stringModel = stringModel
		self._tModel = tModel
		self._raw_score = raw_score
		self._raw_size = raw_size
		self._fitness = fitness
		self._problem_type = problem_type
		self._individual_label = ind_label

	def compute_fitness(self, size_scaler):

		round_up_to = 3

		trainable_count = int(np.sum([K.count_params(p) for p in set(self._tModel.model.trainable_weights)]))
		self._raw_size = trainable_count
		#print("trainable params " + str(trainable_count))
		self.partial_run(cross_validation_ratio, epochs, veborse_train=1)
		metric_score = self._tModel.scores['score_1']
		self._raw_score = metric_score
		#print("metric score " + str(metric_score))

		#Round up to the first 3 digits before computing log
		rounding_scaler = 10**round_up_to
		trainable_count = round(trainable_count/rounding_scaler)*rounding_scaler
		#print("rounded trainable count " + str (trainable_count))


		#For classification estimate the error which is between 0 and 1
		if self._problem_type == 2:
			metric_score = (1 - metric_score)*10 #Multiply by 10 to have a better scaling. I still need to find an appropriate scaling
		else:
			metric_score = metric_score

		size_score = math.log10(trainable_count)

		#print("size score " + str (size_score))
		#print("metric score " + str(metric_score))

		self._fitness = metric_score + size_scaler*size_score
		#print("fitness " + str(self._fitness))

	def partial_run(self, cross_validation_ratio, epochs=20, verbose_data=0, veborse_train=0):
		
		self._tModel.load_data(verbose=verbose_data, cross_validation_ratio=0.2)

		if verbose_data == 1:
			self._tModel.print_data()

		self._tModel.epochs = epochs
		self._tModel.train_model(learningRate_scheduler=lrate, verbose=veborse_train)

		self._tModel.evaluate_model(cross_validation=True)
		cScores = self._tModel.scores
		#print(cScores)

		"""
		self._tModel.evaluate_model(cross_validation=False)
		cScores = self._tModel.scores
		#print(cScores)
		"""

	def __str__(self):

		#self._tModel.model.summary()
		print(self._stringModel)

		return "<Individual(label = '%s' fitness = '%s', raw_score = '%s', raw_size = '%s)>" % (self._individual_label, self._fitness, self._raw_score, self._raw_size)

	#property definition

	@property
	def tModel(self):
		return self._tModel

	@tModel.setter
	def tModel(self, tModel):
		self._tModel = tModel

	@property
	def raw_score(self):
		return self._raw_score

	@raw_score.setter
	def raw_score(self, raw_score):
		self._raw_score = raw_score

	@property
	def raw_size(self):
		return self._raw_size

	@raw_size.setter
	def raw_size(self, raw_size):
		self._raw_size = raw_size

	@property
	def fitness(self):
		return self._fitness

	@fitness.setter
	def fitness(self, fitness):
		self._fitness = fitness

	@property
	def stringModel(self):
		return self._stringModel

	@stringModel.setter
	def stringModel(self, stringModel):
		self._stringModel = stringModel

	@property
	def problem_type(self):
		return self._problem_type

	@problem_type.setter
	def problem_type(self, problem_type):
		self._problem_type = problem_type

	@property
	def individual_label(self):
		return self._individual_label

	@individual_label.setter
	def individual_label(self, individual_label):
		self._individual_label = individual_label


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


def generate_model(model=None, prev_component=Layers.Empty, next_component=Layers.Empty, max_layers=64, more_layers_prob=0.8):
	"""Iteratively and randomly generate a model"""

	layer_count = 0
	success = False

	if model == None:
		model = list()

	while True:

		curr_component = random.choice(ann_building_rules[prev_component])

		#Perturbate param layer
		if curr_component == Layers.PerturbateParam:
			ann_building_rules[Layers.PerturbateParam] = ann_building_rules[prev_component]
		elif curr_component == Layers.Dropout: #Dropout layer
			ann_building_rules[Layers.Dropout] = ann_building_rules[prev_component].copy()
			ann_building_rules[Layers.Dropout].remove(Layers.Dropout)

		rndm = random.random()
		more_layers = (rndm <= more_layers_prob)

		#Keep adding more layers
		if more_layers == False:
			#Is this layer good for ending?
			if next_component == Layers.Empty or next_component in ann_building_rules[curr_component]:
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


def initial_population(pop_size, problem_type, architecture_type, number_classes=2, more_layers_prob=0.5, cross_validation=0):

	population = []

	for i in range(pop_size):
		model_genotype, success = generate_model(more_layers_prob=0.5, prev_component=architecture_type)

		#Generate first layer
		layer_first = generate_layer(architecture_type)

		#Last layer is always FC
		if problem_type == 1:
			layer_last = [Layers.FullyConnected, 1, 4, 0, 0, 0, 0, 0]
		else:
			layer_last = [Layers.FullyConnected, number_classes, 3, 0, 0, 0, 0, 0]

		model_genotype.append(layer_last)
		model_genotype = [layer_first] + model_genotype

		individual = Individual(i, problem_type, model_genotype)

		population.append(individual)

	return population


def tournament_selection(subpopulation):

	if len(subpopulation) < 2:
		print("At least two individuals are required")
		return None
	else:
		most_fit = subpopulation[0]

	for index in range(1, len(subpopulation)):

		individual = subpopulation[index]
		if individual.fitness < most_fit.fitness:
			most_fit = individual

	return most_fit


def population_crossover(parent_pool):

	print("population_crossover")



def individual_crossover(parent1, parent2):

	print("individual_crossover")







