import CMAPSAuxFunctions
import nn_evolutionary
import fetch_to_keras

from data_handler_MNIST import MNISTDataHandler
from tunable_model import SequenceTunableModelRegression
from CMAPSAuxFunctions import TrainValTensorBoard
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from tunable_model import SequenceTunableModelRegression
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

	models = []


	for i in range(10):
		model, success = nn_evolutionary.generate_model(more_layers_prob=0.5, prev_component=architecture_type)

		#Generate first layer
		layer_first = nn_evolutionary.generate_layer(architecture_type)

		#Last layer is always FC
		if problem_type == 1:
			layer_last = [Layers.FullyConnected, 1, 4, 0, 0, 0, 0, 0]
		else:
			layer_last = [Layers.FullyConnected, number_classes, 3, 0, 0, 0, 0, 0]

		model.append(layer_last)
		model_full = [layer_first] + model

		models.append(model_full)

	#Test using mnist
	dHandler_mnist = MNISTDataHandler()

	for i in range(len(models)):

		model_genotype = models[i]
		print("Model run " + str(i))
		print(model_genotype)
		partial_run(model_genotype, problem_type, input_shape, dHandler_mnist, cross_val, i, 5)	




main()