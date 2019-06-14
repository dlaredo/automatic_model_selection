import random
import math
import copy
import logging

import numpy as np

from keras import backend as K

import ann_encoding_rules
from ann_encoding_rules import Layers, LayerCharacteristics, ann_building_rules, activations


class Individual():


	def __init__(self, ind_label, problem_type, stringModel, used_activations, tModel = None, raw_score = 0, raw_size = 0, fitness = 0):
		self._stringModel = stringModel
		self._tModel = tModel
		self._raw_score = raw_score
		self._normalized_score = raw_score
		self._raw_size = raw_size
		self._fitness = fitness
		self._problem_type = problem_type
		self._individual_label = ind_label
		self._used_activations = used_activations
		self._checksum_vector = np.zeros(1)


	def compute_raw_scores(self, epochs, cross_validation_ratio, verbose=0, unroll=False, learningRate_scheduler=None):

		trainable_count = int(np.sum([K.count_params(p) for p in set(self._tModel.model.trainable_weights)]))
		self._raw_size = trainable_count

		self.partial_run(cross_validation_ratio, epochs, verbose=verbose, unroll=unroll, learningRate_scheduler=learningRate_scheduler)
		metric_score = self._tModel.scores['score_1']
		self._raw_score = metric_score
		self._normalized_score = metric_score

		
	def compute_fitness(self, size_scaler):

		round_up_to = 3

		#Round up to the first 3 digits before computing log
		rounding_scaler = 10**round_up_to

		if self._raw_size > rounding_scaler:
			trainable_count = round(self._raw_size/rounding_scaler)*rounding_scaler
		else:
			trainable_count = self._raw_size
		
		scaled_score = self.normalized_score
		
		#For classification estimate the error which is between 0 and 1
		if self._problem_type == 2:
			metric_score = (1 - scaled_score)*10 #Multiply by 10 to have a better scaling. I still need to find an appropriate scaling
		else:
			metric_score = scaled_score*10 #Multiply by 10 to have a better scaling. I still need to find an appropiate scaling

		size_score = math.log10(trainable_count)

		metric_scaler = 1-size_scaler
		print("metric_scaler %f"%metric_scaler)
		print("size scaler %f"%size_scaler)

		#Scalarization of multiobjective version of the fitness function
		self._fitness = metric_scaler*metric_score + size_scaler*size_score


	def compute_checksum_vector(self):

		self._checksum_vector = np.zeros(len(self._stringModel[0]))

		for layer in self._stringModel:

			layer_type = layer[0]
			self._checksum_vector[0] = self._checksum_vector[0] + layer_type.value

			useful_components = ann_encoding_rules.useful_components_by_layer[layer_type]

			for index in useful_components:
				self._checksum_vector[index] = self._checksum_vector[index]+layer[index]


	def partial_run(self, cross_validation_ratio=0.2, epochs=20, verbose=0, unroll=False, learningRate_scheduler=None):

		"""
		print(self._tModel.data_handler)
		print("Data loaded?")
		print(self._tModel.data_loaded)
		"""
		
		self._tModel.load_data(verbose=1, cross_validation_ratio=cross_validation_ratio, unroll=unroll)

		"""
		print("Data loaded?")
		print(self._tModel.data_loaded)
		"""

		self._tModel.epochs = epochs
		self._tModel.train_model(learningRate_scheduler=learningRate_scheduler, verbose=verbose)

		self._tModel.evaluate_model(cross_validation=True)
		cScores = self._tModel.scores
		#print(cScores)

		"""
		self._tModel.evaluate_model(cross_validation=False)
		cScores = self._tModel.scores
		#print(cScores)
		"""

	def __str__(self):

		str_repr1 = "\n\nString Model\n" + str(self._stringModel) + "\n"
		str_repr2 = "<Individual(label = '%s' fitness = %.4f, raw_score = %.4f, normalized_score = %.4f, raw_size = %d)>" % (self._individual_label, self._fitness, self._raw_score, self._normalized_score,  self._raw_size)
		str_repr3 = "\nChecksum vector: " + str(self._checksum_vector)

		str_repr =  str_repr1 + str_repr2 + str_repr3

		return str_repr

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
	def normalized_score(self):
		return self._normalized_score

	@normalized_score.setter
	def normalized_score(self, normalized_score):
		self._normalized_score = normalized_score

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

	@property
	def used_activations(self):
		return self._used_activations

	@used_activations.setter
	def used_activations(self, used_activations):
		self._used_activations = used_activations

	@property
	def checksum_vector(self):
		return self._checksum_vector

	@checksum_vector.setter
	def checksum_vector(self, checksum_vector):
		self._checksum_vector = checksum_vector


def generate_model(model=None, prev_component=Layers.Empty, next_component=Layers.Empty, max_layers=64, more_layers_prob=0.7, used_activations = {}):
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
		rndm = 1 - math.sqrt(1-rndm) #Inverse transformation of -2x + 2 which is the probability distribution we want to follow.
		more_layers = (rndm <= more_layers_prob)

		#Keep adding more layers
		if more_layers == False:
			#Is this layer good for ending?
			if next_component == Layers.Empty or next_component in ann_building_rules[curr_component]:
				layer = ann_encoding_rules.generate_layer(curr_component, used_activations)
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
			layer = ann_encoding_rules.generate_layer(curr_component, used_activations)
			model.append(layer)
			prev_component = curr_component

	return model, success


def initial_population(pop_size, problem_type, architecture_type, number_classes=2, more_layers_prob=0.7):

	population = []

	for i in range(pop_size):

		used_activations = {}

		model_genotype, success = generate_model(more_layers_prob=more_layers_prob, prev_component=architecture_type, used_activations=used_activations)

		#Generate first layer
		layer_first = ann_encoding_rules.generate_layer(architecture_type, used_activations)

		#Last layer is always FC
		if problem_type == 1:
			layer_last = [Layers.FullyConnected, 1, 4, 0, 0, 0, 0, 0]
		else:
			layer_last = [Layers.FullyConnected, number_classes, 3, 0, 0, 0, 0, 0]

		model_genotype.append(layer_last)
		model_genotype = [layer_first] + model_genotype

		individual = Individual(i, problem_type, model_genotype, used_activations)

		population.append(individual)

	return population


def mutation(offsprings, mutation_ratio):

	for individual in offsprings:

		mutation_probability = random.random()
		logging.info("Mutation probability " + str(mutation_probability))
		if mutation_probability < mutation_ratio:

			#pick a layer randomly
			len_model = len(individual.stringModel)
			random_layer_index = np.random.randint(len_model-1)  #Last layer can not be modified

			logging.info("\nInidividual before mutation")
			logging.info(individual)
			logging.info("\nLayer number " + str(random_layer_index))
			logging.info(individual.stringModel[random_layer_index])

			individual.stringModel = layer_based_mutation(individual.stringModel, random_layer_index)

			logging.info("\nInidividual after mutation")
			logging.info(individual)


def layer_based_mutation(stringModel, layer_index, logger=True):
	"""For a given layer, perform a mutation that will effectively affect the layer"""

	layer = stringModel[layer_index]
	layer_next = stringModel[layer_index+1]
	layer_type = layer[0]
	layer_type_next = layer_next[0]
	characteristic = 0
	stringModelCopy = []

	#Randomly select a characteristic from the layer that effectively affects the layer

	if layer_type == Layers.FullyConnected:
		characteristic = random.choice([LayerCharacteristics.NeuronsNumber.value, LayerCharacteristics.ActivationType.value, LayerCharacteristics.DropoutRate.value])

	elif layer_type == Layers.Convolutional:
		characteristic = random.choice([LayerCharacteristics.ActivationType.value, LayerCharacteristics.FilterSizeCNN.value, LayerCharacteristics.KernelSizeCNN.value, 
			LayerCharacteristics.StrideCNN.value, LayerCharacteristics.DropoutRate.value])

	elif layer_type == Layers.Pooling:
		characteristic = LayerCharacteristics.PoolingSize.value

	elif layer_type == Layers.Recurrent:
		characteristic = random.choice([LayerCharacteristics.NeuronsNumber.value, LayerCharacteristics.ActivationType.value, LayerCharacteristics.DropoutRate.value])

	elif layer_type == Layers.Dropout:
		characteristic = LayerCharacteristics.DropoutRate.value

	else:
		characteristic = -1

	characteristic = LayerCharacteristics(characteristic)
	value = ann_encoding_rules.generate_characteristic(layer, characteristic)

	if logger == True:
		logging.info("Choosen characteristic " + str(characteristic))
		logging.info("Selected value " + str(value))

	#If valid layer, then generate 
	if characteristic != LayerCharacteristics.DropoutRate and value != -1:
		layer[characteristic.value] = value

		if characteristic == LayerCharacteristics.ActivationType: #For layer type, rectify entire model with the new activation for layer of type layer_type
			activation = value
			rectify_activations_by_layer_type(stringModel, layer_type, activation)

	elif characteristic == LayerCharacteristics.DropoutRate and value != -1:  #For dropout
		if layer_type != Layers.Dropout:

			if layer_type_next == Layers.Dropout:
				layer_next[LayerCharacteristics.DropoutRate.value] = value
			else:
				stringModelCopy = stringModel[:layer_index+1]

				dropOutLayer = [Layers.Dropout, 0, 0, 0, 0, 0, 0, 0]
				dropOutLayer[LayerCharacteristics.DropoutRate.value] = value

				stringModelCopy.append(dropOutLayer)
				stringModelCopy.extend(stringModel[layer_index+1:])

				stringModel = copy.deepcopy(stringModelCopy)
		else:
			layer[LayerCharacteristics.DropoutRate.value] = value

	return stringModel


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


def binary_tournament_selection(population):

	parent_pool = []

	for index in range(len(population)//2):

		if population[index*2].fitness < population[index*2+1].fitness:
			parent_pool.append(population[index*2])
		else:
			parent_pool.append(population[index*2+1])

	return parent_pool


def rectify_activations_by_layer_type(stringModel, layer_type, activation):
	"""Given a layer type, apply the same activation function to all similar layers"""

	for i in range(len(stringModel)-1):

		layer = stringModel[i]
		if layer[0] == layer_type:
			layer[2] = activation


def rectify_activations_offspring(stringModel):
	"""Rectify the model so that all the similar layers have the same activation"""

	used_activations = {}

	for i in range(len(stringModel)-1):  #Last layer is disregarded as it sould not be changed

		layer = stringModel[i]

		layer_type = layer[0]
		activation = layer[2]

		if layer_type  in used_activations:
			layer[2] = used_activations[layer_type]
		else:
			used_activations[layer_type] = activation

	return used_activations


def population_crossover(parent_pool, max_layers=3, logger=False):

	pop_size = len(parent_pool)//2
	problem_type = parent_pool[0].problem_type
	offsprings = []
	i = 0

	for index in range(pop_size):

		parent1 = parent_pool[index*2]
		parent2 = parent_pool[index*2+1]

		point11, point12, point21, point22, success = two_point_crossover(parent1, parent2, max_layers)

		if logger == True:

			logging.info("\nParents\n")
			logging.info(parent1.stringModel)
			logging.info(parent2.stringModel)
			logging.info("Points: {} {} {} {}, success: {}".format(point11, point12, point21, point22, success))

		#If a valid model was created then proceed
		if success == True:

			#Dont include layers in parent 1
			offspring_stringModel = parent1.stringModel[:point11]
			offspring_stringModel.extend(parent2.stringModel[point21:point22+1])
			offspring_stringModel.extend(parent1.stringModel[point12+1:])

			#Perform deep copy to avoid cross references
			offspring_stringModel = copy.deepcopy(offspring_stringModel)

			used_activations = rectify_activations_offspring(offspring_stringModel)

			if logger == True:
				logging.info("Offspring\n")
				logging.info(offspring_stringModel)

			offspring = Individual(pop_size+i, problem_type, offspring_stringModel, used_activations)
			offsprings.append(offspring)
			i = i+1

	return offsprings


def two_point_crossover(parent1, parent2, max_layers, max_attempts=5, logger=False):

	stringModel1 = parent1.stringModel
	len_model1 = len(stringModel1)-1  #Len model-1 because the last layer can not be moved

	attempts = 0
	success = False

	while attempts < max_attempts:

		temp = 0
		attempts = attempts + 1
		compatible_substructures = []

		#Choose a random point to do the two point crossover
		point11 = np.random.randint(len_model1)
		point12 = np.random.randint(len_model1)
		point21 = -1
		point22 = -1

		if point11 > point12:
			temp = point11
			point11 = point12
			point12 = temp
		elif point11 == point12:  #Should it go all the way to the end of the string or just stay there?
			point12 = len_model1-1  #Go to the last layer (excepting the output layer)
		else:
			pass

		first_layer = True if point11 == 0 else False

		if first_layer == True:
			layer_prev = stringModel1[point11]
		else:
			layer_prev = stringModel1[point11-1]

		layer_next = stringModel1[point12+1]

		compatible_previuos, compatible_next = find_match(parent2, layer_prev, layer_next, first_layer, max_layers)

		if compatible_next != [] or compatible_previuos != []:  #If there are compatible layers, proceed, otherwise look for other crossover points

			#Make pairs of possible substructures
			for i  in compatible_previuos:
				for j in compatible_next:

					if j-i < max_layers and j-i >= 0:
						compatible_substructures.append((i,j)) 

			if compatible_substructures != []:

				k = np.random.randint(len(compatible_substructures))
				chosen_substructure = compatible_substructures[k]
				point21 = chosen_substructure[0]
				point22 = chosen_substructure[1]
				success = True
				break

	return (point11, point12, point21, point22, success)


def find_match(parent, layer_prev, layer_next, first_layer, max_layers):
	"""Try to find compatible layers according to the points chosen by the parent1"""

	stringModel = parent.stringModel
	len_model = len(stringModel)

	point11 = 0
	point12 = 0

	compatible_previuos = []
	compatible_next = []

	for i in range(len_model-1):  #Dismiss last layer

		layer = stringModel[i]

		#Check forward compatibility
		if first_layer == True: 
			if layer[0] == layer_prev[0]:
				compatible_previuos.append(i)
		else:
			compatible_layers = ann_building_rules[layer_prev[0]]

			if layer[0] in compatible_layers:
				compatible_previuos.append(i)

		#Check backward compatibility
		compatible_layers = ann_building_rules[layer[0]]

		if layer_next[0] in compatible_layers:
			compatible_next.append(i)

	return compatible_previuos, compatible_next


def generation_similar(population, max_similar, similar_threshold=0.9, logger=False):
	#Compute distances between elements and remove those that are very similar

	new_pop = []
	len_pop = len(population)
	pairs = []
	distances = {}
	max_distance = 0
	max_pair = None
	similar = 0
	generation_similar = False

	for i in range(len_pop):
		for j in range(len_pop):
			if j > i:
				pairs.append((i,j))

	for pair in pairs:
		i = pair[0]
		j = pair[1]

		"""
		distance = population[i].checksum_vector - population[j].checksum_vector
		print(distance)
		distance_norm = np.linalg.norm(distance, 2)
		"""

		distance_norm = distance_between_models(population[i].stringModel, population[j].stringModel)
		distances[pair] = distance_norm

		if distance_norm > max_distance:
			max_distance = distance_norm
			max_pair = pair

	#If max distance is zero, then all the elements in the generation are equal
	if max_distance == 0:
		generation_similar = True
	else:
		#Normalize distances and see how many are greater than threshold
		for key in distances:
			normalized_distance = distances[key]/max_distance
			distances[key] = normalized_distance

			if normalized_distance < similar_threshold and key != max_pair:
				similar = similar + 1

	if similar > max_similar:
		generation_similar = True

	if logger == True:
		logging.info("similar = " + str(similar))
		logging.info(distances)

	return generation_similar


def distance_between_models(stringModel1, stringModel2):
	"""Compute the similarity between two given models"""

	len_model1 = len(stringModel1)
	len_model2 = len(stringModel2)

	len_layer = len(stringModel1[0])

	layer_distance = np.zeros(len_layer)

	distance = 0

	if len_model1 > len_model2:

		for i in range(len_model2-1):
			layer_distance[0] = stringModel1[i][0].value - stringModel2[i][0].value

			for j in range(1, len_layer):
				layer_distance[j] = stringModel1[i][j] - stringModel2[i][j]

			distance += np.linalg.norm(layer_distance, 2)

		for i in range(len_model2-1, len_model1-1):
			layer_distance[0] = stringModel1[i][0].value

			for j in range(1, len_layer):
				layer_distance[j] = stringModel1[i][j]

			distance += np.linalg.norm(layer_distance, 2)

	else:

		for i in range(len_model1-1):
			layer_distance[0] = stringModel2[i][0].value - stringModel1[i][0].value

			for j in range(1, len_layer):
				layer_distance[j] = stringModel2[i][j] - stringModel1[i][j]

			distance += np.linalg.norm(layer_distance, 2)

		for i in range(len_model1-1, len_model2-1):
			layer_distance[0] = stringModel2[i][0].value

			for j in range(1, len_layer):
				layer_distance[j] = stringModel2[i][j]

			distance += np.linalg.norm(layer_distance, 2)	

	return distance	








