from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import random

"""Layer types

	LBlock:1
	FC:2
	Conv:3
	Pooling:4
	RNN:5
	PerturbateParam:6
	Empty:7

"""

"""Layer tuple

	0: Layer type
	1: Neuron number
	2: Activation function (0:sigmoid, 1:tanh, 2:relu)
	3: Filters for cnn
	4: Kernel_size for cnn (just one for squared kernels)
	5: Strides for cnn

"""

ann_building_rules = {
	
					1:(1, 2, 3, 4, 5, 6, 7),
					2:(2, 6, 7),
					3:(2, 3, 4, 6, 7),
					4:(2, 3, 6, 7),
					5:(2, 6, 7),
					6:(1, 2, 3, 4, 5, 6, 7),
					7:(1, 2, 3, 4, 5, 6, 7)
}

nodes = {
	
	1:("LBlock", 2),
	2:("FC", 2),
	3:("Conv", 2),
	4:("Pool", 2),
	5:("RNN", 2),
	6:("PerturbateParam", 0),
	7:("Empty", 0)

}

def generate_model(model=None, first_component=7, last_component=7, max_layers=64, more_layers_prob=0.8):
	"""Iteratively and randomly generate a model"""

	layer_count = 0
	success = False

	if model == None:
		model = list()



	while True:
		next_layer = random.choice(ann_building_rules[first_component])

		more_layers = random.random() <= more_layers_prob

		#Keep adding more layers
		if more_layers == False:
			#Is this layer good for ending?
			if last_component == 7 or last_component == next_layer:
				layer = generate_layer(next_layer)
				model.append(layer)
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
			layer = generate_layer(next_layer)
			model.append(layer)

	return model, success


def generate_layer(layer_type):
	"""Given a layer type, return the layer params"""

	layer = [layer_type, 0, 0, 0, 0, 0]
	layer[1] = 8*random.randint(1,129) #Generate a random number of neurons which is a multiple of 8 up to 1024 neurons
	layer[2] = random.randint(0,3) if layer_type != 5 else 2
	layer[3] = 8*random.randint(1,64)
	layer[4] = 2**random.randint(1,8)
	layer[5] = random.randint(1,4)

	return layer


	"""
	if layer_type == 2: #FC layer
	elif layer_type == 3: #Conv layer
	elif layer_type == 4: #Pooling layer
	elif layer_type == 5: #RNN layer
	else:
	"""

def main():

	model, success = generate_model()
	print(success)
	print(model)

main()
















