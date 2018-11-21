import nn_evolutionary
import fetch_to_keras
import random
import datetime
import logging

from data_handler_MNIST import MNISTDataHandler
from tunable_model import SequenceTunableModelRegression
from CMAPSAuxFunctions import TrainValTensorBoard
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from ann_encoding_rules import Layers



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
	tModel.load_data(verbose=1, cross_validation_ratio=0.2)
	tModel.print_data()

	tModel.epochs = epochs
	tModel.train_model(learningRate_scheduler=lrate, verbose=1)

	tModel.evaluate_model(cross_validation=True)

	cScores = tModel.scores
	print(cScores)

	tModel.evaluate_model(cross_validation=False)

	cScores = tModel.scores
	print(cScores)


def print_pop(parent_pool):

	for ind in parent_pool:
		print(ind)



def main():
	"""Input can be of 3 types, ANN (1), CNN (2) or RNN (3)"""

	architecture_type = Layers.FullyConnected
	problem_type = 2  #1 for regression, 2 for classification
	number_classes = 10 #If regression applies, number of classes
	input_shape = (784,)
	cross_val = 0.2
	generations = 0
	pop_size = 5
	tournament_size = 4
	binary_selection = True
	mutation_ratio = 0.8

	t = datetime.datetime.now()

	parent_pop = []
	elite_archive = []

	best_model = nn_evolutionary.Individual(pop_size*2, problem_type, [], [], fitness=10**8)  #Big score for the first comparisson
	worst_model = nn_evolutionary.Individual(pop_size*2+1, problem_type, [], [], fitness=0)  #Big score for the first comparisson
	worst_index = 0

	logging.basicConfig(filename='logs/nn_evolution_' + t.strftime('%m%d%Y%H%M%S') + '.log', level=logging.INFO, 
		format='%(levelname)s:%(threadName)s:%(asctime)s:%(filename)s:%(funcName)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')


	#Test using mnist
	dHandler_mnist = MNISTDataHandler()

	population = nn_evolutionary.initial_population(pop_size, problem_type, architecture_type, number_classes=number_classes,
		more_layers_prob=0.8, cross_validation=cross_val)


	for i in range(generations):
		
		count = 0
		worst_index = 0
		parent_pool = []
		#min_score = 10**8 #Big score for the first comparisson
		offsprings = []

		indices = list(range(pop_size))

		fetch_to_keras.population_to_keras(population, input_shape, dHandler_mnist)

		print("Empty best model")
		print(best_model)

		#Evaluate population
		for individual in population:
			individual.tModel.model.summary()
			individual.compute_fitness(size_scaler=1)
			print(individual)

			#Get generation best
			if individual.fitness < best_model.fitness:
				best_model = individual

			#Replace worst with previous best
			if individual.fitness > worst_model.fitness:
				worst_model = individual
				worst_index = count

			count = count+1


		print("Generation Best model")
		print(best_model)

		print("Generation worst model")
		print(worst_model)

		print_pop(population)

		if i > 0: #At least one generation so to have one best model

			previous_best = elite_archive[-1]
			if previous_best.fitness < worst_model.fitness:
				population[worst_index] = previous_best
				print("Worst replaced")


		print_pop(population)
		elite_archive.append(best_model)

		offsprings_complete = False
		print("offsprings")
		print(offsprings)

		#select 2*(n-1) individuals for crossover, elitism implemented
		offspring_pop_size = 0
		while pop_size-offspring_pop_size > 0:

			count = 0
			parents_pool_required = 2*(pop_size-offspring_pop_size)

			while count < parents_pool_required:

				print("binary selection? " + str(binary_selection))

				if binary_selection == True:
					random.shuffle(indices)
					print(indices)
					indices_tournament = indices[:tournament_size]
					print(indices_tournament)
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

			print(len(parent_pool))
			print_pop(parent_pool)
			#print(parent_pool)
			offsprings = nn_evolutionary.population_crossover(parent_pool)
			offspring_pop_size = len(offsprings)

		print(offsprings)
		print()
		print("mutation")
		nn_evolutionary.mutation(offsprings, mutation_ratio)
		print("mutated offsprings")
		print_pop(offsprings)
		#print("population")
		parent_pop = population
		population = []
		#print(offsprings)
		population = offsprings
		offsprings = []
		#print(parent_pop)
		#print(population)




main()













