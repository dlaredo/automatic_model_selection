import nn_evolutionary
import fetch_to_keras
import random
import logging

#import ray

import numpy as np

from keras import backend as K


class Configuration():

	def __init__(self, architecture_type, problem_type, input_shape, output_shape, pop_size, tournament_size, max_similar, size_scaler=1, epochs=1, cross_val=0.2, more_layers_prob=0.5, 
		     max_generations=1, binary_selection=True, mutation_ratio=0.4, similarity_threshold=0.9, verbose_individuals=False, show_model=False, verbose_training=0):
		
		self._architecture_type = architecture_type
		self._problem_type = problem_type  #1 for regression, 2 for classification
		self._input_shape = input_shape
		self._output_shape = output_shape #If regression applies, number of classes
		self._cross_val = cross_val
		self._more_layers_prob = more_layers_prob
		self._max_generations = max_generations
		self._pop_size = pop_size
		self._tournament_size = tournament_size
		self._binary_selection = binary_selection
		self._mutation_ratio = mutation_ratio
		self._similarity_threshold = similarity_threshold
		self._max_similar = max_similar
		self._epochs = epochs
		self._size_scaler = size_scaler
		self._verbose_individuals = verbose_individuals
		self._verbose_training = verbose_training
		self._show_model = show_model

	@property
	def architecture_type(self):
		return self._architecture_type

	@architecture_type.setter
	def architecture_type(self, architecture_type):
		self._architecture_type = architecture_type

	@property
	def problem_type(self):
		return self._problem_type

	@problem_type.setter
	def problem_type(self, problem_type):
		self._problem_type = problem_type

	@property
	def input_shape(self):
		return self._input_shape

	@input_shape.setter
	def input_shape(self, input_shape):
		self._input_shape = input_shape

	@property
	def output_shape(self):
		return self._output_shape

	@output_shape.setter
	def output_shape(self, output_shape):
		self._output_shape = output_shape

	@property
	def epochs(self):
		return self._epochs

	@epochs.setter
	def epochs(self, epochs):
		self._epochs = epochs

	@property
	def cross_val(self):
		return self._cross_val

	@cross_val.setter
	def cross_val(self, cross_val):
		self._cross_val = cross_val

	@property
	def more_layers_prob(self):
		return self._more_layers_prob

	@more_layers_prob.setter
	def more_layers_prob(self, more_layers_prob):
		self._more_layers_prob = more_layers_prob

	@property
	def size_scaler(self):
		return self._size_scaler

	@size_scaler.setter
	def size_scaler(self, size_scaler):
		self._size_scaler = size_scaler

	@property
	def max_generations(self):
		return self._max_generations

	@max_generations.setter
	def max_generations(self, max_generations):
		self._max_generations = max_generations

	@property
	def pop_size(self):
		return self._pop_size

	@pop_size.setter
	def pop_size(self, pop_size):
		self._pop_size = pop_size

	@property
	def tournament_size(self):
		return self._tournament_size

	@tournament_size.setter
	def tournament_size(self, tournament_size):
		self._tournament_size = tournament_size

	@property
	def binary_selection(self):
		return self._binary_selection

	@binary_selection.setter
	def binary_selection(self, binary_selection):
		self._binary_selection = binary_selection

	@property
	def mutation_ratio(self):
		return self._mutation_ratio

	@mutation_ratio.setter
	def mutation_ratio(self, mutation_ratio):
		self._mutation_ratio = mutation_ratio

	@property
	def max_similar(self):
		return self._max_similar

	@max_similar.setter
	def max_similar(self, max_similar):
		self._max_similar = max_similar

	@property
	def similarity_threshold(self):
		return self._similarity_threshold

	@similarity_threshold.setter
	def similarity_threshold(self, similarity_threshold):
		self._similarity_threshold = similarity_threshold

	@property
	def verbose_individuals(self):
		return self._verbose_individuals

	@verbose_individuals.setter
	def verbose_individuals(self, verbose_individuals):
		self._verbose_individuals = verbose_individuals

	@property
	def verbose_training(self):
		return self._verbose_training

	@verbose_training.setter
	def verbose_training(self, verbose_training):
		self._verbose_training = verbose_training

	@property
	def show_model(self):
		return self._show_model

	@show_model.setter
	def show_model(self, show_model):
		self._show_model = show_model



def evaluate_individual(individual, configuration, data_handler, tModel_scaler, ind_index, unroll, learningRate_scheduler=None):
	"""Given an individual (that only contains the string model) fetch it to keras and evaluate it
	This is done this way to make possible the distribution of the model evaluation. Furthermore
	this allows that we can reset the keras/tensorflow session everytime we evaluate a new model"""

	"""Clear the session to avoid clutter from old models/layers (only useful for small number of models,
		as the number of models grows the session will become more and more cluttered)"""
	K.clear_session()

	#Fetch the individual to keras
	if configuration.show_model:
		print("Fetching model {} to keras".format(ind_index))

	tModel = fetch_to_keras.create_tunable_model(individual.stringModel, individual.problem_type, configuration.input_shape, data_handler, ind_index)
	individual.tModel = tModel

	if tModel_scaler != None:
		tModel.data_handler.data_scaler = None
		tModel.data_scaler = tModel_scaler

	if configuration.show_model:
		print("Evaluating model {}".format(ind_index))
		print(tModel.model.summary())

	"""
	individual.compute_fitness(epochs=configuration.epochs, cross_validation_ratio=configuration.cross_val,
							   size_scaler=configuration.size_scaler, verbose=configuration.verbose_training, unroll=unroll, learningRate_scheduler=learningRate_scheduler)
	"""

	individual.compute_raw_scores(epochs=configuration.epochs, cross_validation_ratio=configuration.cross_val,
				      verbose=configuration.verbose_training, unroll=unroll, learningRate_scheduler=learningRate_scheduler)
	
	individual.individual_label = ind_index



def evaluate_population(population, configuration, data_handler, tModel_scaler, unroll, learningRate_scheduler=None):
	"""Given the population, evaluate it using a framework for deep learning (keras/tensorflow)"""

	best_model = nn_evolutionary.Individual(configuration.pop_size*2, configuration.problem_type, [], [], fitness=10**8)  #Big score for the first comparisson
	worst_model = nn_evolutionary.Individual(configuration.pop_size*2+1, configuration.problem_type, [], [], fitness=0)  #Small score for the first comparisson
	
	count = 0
	worst_index = 0
	pop_size = len(population)
	raw_scores = np.zeros((pop_size))
	normalized_scores = None

	#Compute the raw score for each individual
	for i in range(pop_size):
		individual = population[i]

		evaluate_individual(individual, configuration, data_handler, tModel_scaler, i, unroll,
							learningRate_scheduler=learningRate_scheduler)

		raw_scores[i] = individual.raw_score
		
		#if configuration.verbose_individuals == True:
		#	print("Individual {} score/size/fitness {}/{}/{}".format(i, individual.raw_score, individual.raw_size, individual.fitness))

	#Normalize population
	normalization_factor = np.linalg.norm(raw_scores)
	#normalization_factor = raw_scores.max()
	
	normalized_scores = raw_scores/normalization_factor
	#print(normalization_factor)
	#print(normalized_scores)

	#Compute fitness based on normalized score for each individual
	for i in range(pop_size):

		individual = population[i]

		#For classification we use the accuracy
		if individual.problem_type == 2:
			individual.normalized_score = raw_scores[i]
		else:  #For regression we used normalized RMSE
			individual.normalized_score = normalized_scores[i]

		individual.compute_fitness(configuration.size_scaler)

		if configuration.verbose_individuals == True:
			print("Individual {} score/normalized score/size/fitness {}/{}/{}/{}".format(i, individual.raw_score, individual.normalized_score, individual.raw_size, individual.fitness))
			logging.info("Individual {} score/normalized score/size/fitness {}/{}/{}/{}".format(i, individual.raw_score, individual.normalized_score, individual.raw_size, individual.fitness))
			
		#Get generation best
		if individual.fitness < best_model.fitness:
			best_model = individual

		#Replace worst with previous best
		if individual.fitness > worst_model.fitness:
			worst_model = individual
			worst_index = count


	return best_model, worst_model, worst_index


def evaluate_population2(population, configuration, data_handler, tModel_scaler, best_model, worst_model, unroll, verbose_data):
	"""Given the population, evaluate it using a framework for deep learning ("keras")"""

	count = 0
	worst_index = 0

	#Fetch to keras	
	print("Fetching to keras")
	fetch_to_keras.population_to_keras(population, configuration.input_shape, data_handler, tModel_scaler=tModel_scaler)


	print("Evaluating population")

	#Evaluate population
	for individual in population:
		individual.tModel.model.summary()
		individual.compute_fitness(epochs=configuration.epochs, cross_validation_ratio=configuration.cross_val, size_scaler=configuration.size_scaler, verbose_data=verbose_data, unroll=unroll)
		individual.individual_label = count

		#Get generation best
		if individual.fitness < best_model.fitness:
			best_model = individual

		#Replace worst with previous best
		if individual.fitness > worst_model.fitness:
			worst_model = individual
			worst_index = count

		individual.individual_label = count

		count = count+1

	return best_model, worst_model, worst_index


"""
@ray.remote
def partial_run(model_genotype, problem_type, input_shape, data_handler, cross_validation_ratio, run_number, epochs=20):
	#Function to partially run a model given its genotype and input data
	#This should be run in Ray

	#How to keep the data in a way such that it doesnt create too much overhead

	K.clear_session()  #Clear the previous tensorflow graph
	model = fetch_to_keras.decode_genotype(model_genotype, problem_type, input_shape, 1)

	if model != None:
		model.summary()

	lrate = fetch_to_keras.LearningRateScheduler(CMAPSAuxFunctions.step_decay)

	model = fetch_to_keras.get_compiled_model(model, problem_type, optimizer_params=[])
	tModel = SequenceTunableModelRegression('ModelMNIST_SN_'+str(run_number), model, lib_type='keras', data_handler=data_handler)
	

	#tModel.load_data(verbose=1, cross_validation_ratio=0.2)
	tModel.load_data(verbose=1, cross_validation_ratio=0.2, unroll=True)


	tModel.print_data()

	tModel.epochs = epochs
	tModel.train_model(learningRate_scheduler=lrate, verbose=1)

	tModel.evaluate_model(cross_validation=True)
	cScores = tModel.scores
	print(cScores)
"
"""


def partial_run(model_genotype, problem_type, input_shape, data_handler, cross_validation_ratio, run_number, epochs=20):
	"""Function to partially run a model given its genotype and input data"""
	"""This should be run in Ray"""

	"""How to keep the data in a way such that it doesnt create too much overhead"""

	K.clear_session()  #Clear the previous tensorflow graph
	model = fetch_to_keras.decode_genotype(model_genotype, problem_type, input_shape, 1)

	if model != None:
		model.summary()

	lrate = fetch_to_keras.LearningRateScheduler(CMAPSAuxFunctions.step_decay)

	model = fetch_to_keras.get_compiled_model(model, problem_type, optimizer_params=[])
	tModel = SequenceTunableModelRegression('ModelMNIST_SN_'+str(run_number), model, lib_type='keras', data_handler=data_handler)
	

	#tModel.load_data(verbose=1, cross_validation_ratio=0.2)
	tModel.load_data(verbose=1, cross_validation_ratio=0.2, unroll=True)


	tModel.print_data()

	tModel.epochs = epochs
	tModel.train_model(learningRate_scheduler=lrate, verbose=1)

	tModel.evaluate_model(cross_validation=True)
	cScores = tModel.scores
	print(cScores)


def print_pop(parent_pool, logger=False):

	for ind in parent_pool:
		if logger == False:
			print(ind)
		else:
			logging.info(str(ind))


def run_experiment(configuration, data_handler, experiment_number, unroll=False, learningRate_scheduler=None, tModel_scaler=None):
	"""Run one experiment. An experiment consists of running the evolutionary algorithm for n generations"""

	launch_new_generation = True #First generation is always launched
	experiment_best = None
	generation_count = 0

	#parent_pop = []
	elite_archive = []  #Archive to store the best individuals in each generation

	#Log the information of this experiment
	logging.info("Starting model optimization: Problem type {}, Architecture type {}".format(configuration.problem_type, configuration.architecture_type))
	logging.info("Parameters:")
	logging.info("Input shape: {}, Output shape: {}, cross_val ratio: {}, Generations: {}, Population size: {}, Tournament size: {}, Binary selection: {}, Mutation ratio: {}, Size scaler: {}".format(
		configuration.input_shape, configuration.output_shape, configuration.cross_val, configuration.max_generations, configuration.pop_size, 
		configuration.tournament_size, configuration.binary_selection, configuration.mutation_ratio, configuration.size_scaler))


	population = nn_evolutionary.initial_population(configuration.pop_size, configuration.problem_type, configuration.architecture_type, number_classes=configuration.output_shape,
		more_layers_prob=configuration.more_layers_prob)


	while launch_new_generation == True and generation_count < configuration.max_generations:
		
		count = 0
		worst_index = 0
		parent_pool = []
		offsprings = []

		indices = list(range(configuration.pop_size))

		print("\nGeneration " + str(generation_count+1))
		logging.info("\n\nGeneration " + str(generation_count+1))

		#Fill checksum vector
		for ind in population:
			ind.compute_checksum_vector()

		#If the individuals in the generation are very similar prematurely stop the experiment (Here it is important to address the fact on how to measure similarity)
		generation_similar = nn_evolutionary.generation_similar(population, configuration.max_similar, configuration.similarity_threshold, logger=True)
		launch_new_generation = not generation_similar

		print("launch new")
		print(launch_new_generation)
		print("gen similar")
		print(generation_similar)

		#Assess the fitness of the inidividuals in the population
		best_model, worst_model, worst_index = evaluate_population(population, configuration, data_handler, tModel_scaler,
																   unroll, learningRate_scheduler)

		#Save worst and best models. Also append best model to elite archive
		logging.info("\nPopulation at generation " + str(generation_count+1))
		print_pop(population, logger=True)

		logging.info("\nGeneration Best model")
		logging.info(best_model)

		logging.info("\nGeneration worst model")
		logging.info(worst_model)

		if generation_count > 0: #At least one generation so to have one best model

			previous_best = elite_archive[-1]
			if previous_best.fitness < worst_model.fitness:
				population[worst_index] = previous_best
				logging.info("\nWorst individual replaced")
				logging.info("Poulation after replacing worst with best")
				print_pop(population, logger=True)

		elite_archive.append(best_model)

		#Save global best
		if experiment_best == None:
			experiment_best = best_model
		else:
			if best_model.fitness < experiment_best.fitness:
				experiment_best = best_model

		#Proceed with rest of algorithm
		#offsprings_complete = False
		print("\nGenerating offsprings")
		logging.info("\n\nCrossover\n\n")

		#select 2*(n-1) individuals for crossover, elitism implemented
		offspring_pop_size = 0
		while configuration.pop_size-offspring_pop_size > 0:

			count = 0
			parents_pool_required = 2*(configuration.pop_size-offspring_pop_size)

			logging.info("\nGetting offpsrings with selection: " + 'binary tournament' if configuration.binary_selection == True else 'tournament')

			while count < parents_pool_required:

				if configuration.binary_selection == True:
					random.shuffle(indices)
					indices_tournament = indices[:configuration.tournament_size]

					logging.info(indices)
					logging.info(indices_tournament)
					
					subpopulation = [population[index] for index in indices_tournament]
					selected_individuals = nn_evolutionary.binary_tournament_selection(subpopulation)
					parent_pool.extend(selected_individuals)
					count = len(parent_pool)
				else:
					ind_indices = random.sample(indices, 2)
					subpopulation = [population[index] for index in ind_indices]
					selected_individuals = nn_evolutionary.tournament_selection(subpopulation)
					parent_pool.append(selected_individuals)
					count = len(parent_pool)

			logging.info("\nParent pool. Parent number {}\n".format(len(parent_pool)))
			print_pop(parent_pool, logger=True)

			offsprings = nn_evolutionary.population_crossover(parent_pool, logger=True)
			offspring_pop_size = len(offsprings)

		print("Applying Mutation")
		logging.info("\n\nMutation\n\n")
		nn_evolutionary.mutation(offsprings, configuration.mutation_ratio)

		population = []

		population = offsprings
		offsprings = []

		generation_count =  generation_count + 1

		print("Launch new generation?: " + str(launch_new_generation))
		logging.info("Launch new generation?: " + str(launch_new_generation))

	print("Experiment {} finished".format(experiment_number))
	logging.info("Experiment {} finished".format(experiment_number))
	return experiment_best


def print_best(best_list):

	for i in best_list:
		print(i)
		logging.info(i)













