import random

from ann_encoding_rules import Layers, ann_building_rules, activations



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





