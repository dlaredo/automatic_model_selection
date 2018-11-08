import nn_evolutionary
import fetch_to_keras
import random

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





def main():
	"""Input can be of 3 types, ANN (1), CNN (2) or RNN (3)"""

	architecture_type = Layers.FullyConnected
	problem_type = 2  #1 for regression, 2 for classification
	number_classes = 10 #If regression applies, number of classes
	input_shape = (784,)
	cross_val = 0.2
	generations = 1
	pop_size = 6
	tournament_size = 4


	#Test using mnist
	dHandler_mnist = MNISTDataHandler()

	population = nn_evolutionary.initial_population(pop_size, problem_type, architecture_type, number_classes=number_classes,
		more_layers_prob=0.5, cross_validation=cross_val)


	for i in range(generations):
		
		count = 0
		parent_pool = []

		indices = list(range(pop_size))

		fetch_to_keras.population_to_keras(population, input_shape, dHandler_mnist)

		#Evaluate population
		for individual in population:
			individual.tModel.model.summary()
			individual.compute_fitness(size_scaler=1)
			print(individual)

		#select 2*(n-1) individuals for crossover, elitism implemented
		while count < 2*(pop_size-1):

			ind_indices = random.sample(indices, tournament_size)
			print(ind_indices)

			subpopulation = [population[index] for index in ind_indices]
			#print(subpopulation)
			selected_individual = nn_evolutionary.tournament_selection(subpopulation)
			#print(selected_individual)
			parent_pool.append(selected_individual)
			count = count + 1




main()













