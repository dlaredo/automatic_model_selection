import nn_evolutionary
import fetch_to_keras
import random
import datetime
import logging
import sys
import numpy as np

sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects/data_handlers/')

from tunable_model import SequenceTunableModelRegression
from CMAPSAuxFunctions import TrainValTensorBoard
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from ann_encoding_rules import Layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Data handlers
from data_handler_MNIST import MNISTDataHandler
from data_handler_CMAPS import CMAPSDataHandler



#from data_handlers import data_handler_MNIST


class Configuration():

	def __init__(self, architecture_type, problem_type, input_shape, output_shape, pop_size, tournament_size, max_similar, size_scaler=1, epochs=1, cross_val=0.2, more_layers_prob=0.5, 
		max_generations=1, binary_selection=True, mutation_ratio=0.4, similarity_threshold=0.9):
		
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



def partial_run(model_genotype, problem_type, input_shape, data_handler, cross_validation_ratio, run_number, epochs=20):
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

	"""
	tModel.evaluate_model(cross_validation=False)
	cScores = tModel.scores
	print(cScores)
	"""


def print_pop(parent_pool, logger=False):

	for ind in parent_pool:
		if logger == False:
			print(ind)
		else:
			logging.info(str(ind))


def run_experiment(configuration, data_handler, experiment_number, unroll=False, verbose_data=0, tModel_scaler=None):
	"""Run one experiment"""
	launch_new_generation = True #First generation is always launched
	experiment_best = None
	generation_count = 0

	parent_pop = []
	elite_archive = []

	best_model = nn_evolutionary.Individual(configuration.pop_size*2, configuration.problem_type, [], [], fitness=10**8)  #Big score for the first comparisson
	worst_model = nn_evolutionary.Individual(configuration.pop_size*2+1, configuration.problem_type, [], [], fitness=0)  #Big score for the first comparisson
	worst_index = 0

	logging.info("Starting model optimization: Problem type {}, Architecture type {}".format(configuration.problem_type, configuration.architecture_type))
	logging.info("Parameters:")
	logging.info("Input shape: {}, Output shape: {}, cross_val ratio: {}, Generations: {}, Population size: {}, Tournament size: {}, Binary selection: {}, Mutation ratio: {}".format(
		configuration.input_shape, configuration.output_shape, configuration.cross_val, configuration.max_generations, configuration.pop_size, 
		configuration.tournament_size, configuration.binary_selection, configuration.mutation_ratio))


	population = nn_evolutionary.initial_population(configuration.pop_size, configuration.problem_type, configuration.architecture_type, number_classes=configuration.output_shape,
		more_layers_prob=configuration.more_layers_prob, cross_validation=configuration.cross_val)


	while launch_new_generation == True and generation_count < configuration.max_generations:
		
		count = 0
		worst_index = 0
		parent_pool = []
		offsprings = []

		indices = list(range(configuration.pop_size))

		print("\nGeneration " + str(generation_count+1))
		logging.info("\n\nGeneration " + str(generation_count+1))

		#Remove those individuals that are very similar
		for ind in population:
			ind.compute_checksum_vector()

		launch_new_generation = nn_evolutionary.launch_new_generation(population, configuration.max_similar, configuration.similarity_threshold, logger=True)

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
		offsprings_complete = False
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


def cmaps_dhandler():

	#Selected as per CNN paper
	features = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 
	'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
	selected_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
	selected_features = list(features[i] for i in selected_indices-1)
	data_folder = '../CMAPSSData'

	window_size = 25
	window_stride = 1
	max_rul = 130

	dHandler_cmaps = CMAPSDataHandler(data_folder, 1, selected_features, max_rul, window_size, window_stride)

	input_shape = (len(selected_features)*window_size, )

	return dHandler_cmaps, input_shape


def main():
	"""Input can be of 3 types, ANN (1), CNN (2) or RNN (3)"""
	architecture_type = Layers.FullyConnected
	problem_type = 1  #1 for regression, 2 for classification
	output_shape = 1 #If regression applies, number of classes
	input_shape = (784,)
	cross_val = 0.2
	pop_size = 5
	tournament_size = 4
	max_similar = 3
	total_experiments = 5
	new_experiment = True
	count_experiments = 0
	unroll = True

	global_best_list = []
	global_best = None

	min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

	t = datetime.datetime.now()

	logging.basicConfig(filename='logs/nn_evolution_' + t.strftime('%m%d%Y%H%M%S') + '.log', level=logging.INFO, 
		format='%(levelname)s:%(threadName)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')

	#Test using mnist
	dhandler_cmaps, input_shape = cmaps_dhandler()

	config = Configuration(architecture_type, problem_type, input_shape, output_shape, pop_size, tournament_size, max_similar, epochs=1, cross_val=0.2, size_scaler=1,
		max_generations=10, binary_selection=True, mutation_ratio=0.4, similarity_threshold=0.2, more_layers_prob=0.8)

	while count_experiments < total_experiments:
		print("Launching experiment {}".format(count_experiments+1))
		logging.info("Launching experiment {}".format(count_experiments+1))
		best = run_experiment(config, dhandler_cmaps, count_experiments + 1, unroll=unroll, verbose_data=1, tModel_scaler=min_max_scaler)

		global_best_list.append(best)

		if global_best == None:
			global_best = best
		else:
			if best.fitness < global_best.fitness:
				global_best = best

		count_experiments =  count_experiments + 1

	print("Global best list\n")
	print(global_best_list)
	print("Global best is\n")
	print(global_best)

main()













