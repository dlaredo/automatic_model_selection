from enum import Enum

class Layers(Enum):
	FullyConnected = 1
	Convolutional = 2
	Pooling = 3
	Recurrent = 4
	Dropout = 5
	PerturbateParam = 6
	Empty = 7


ann_building_rules = {
	
					Layers.FullyConnected:[Layers.FullyConnected, Layers.Dropout],
					Layers.Convolutional:[Layers.FullyConnected, Layers.Convolutional, Layers.Pooling, Layers.Dropout],
					Layers.Pooling:[Layers.FullyConnected, Layers.Convolutional],
					Layers.Recurrent:[Layers.FullyConnected, Layers.Convolutional],
					Layers.Dropout:[Layers.FullyConnected, Layers.Convolutional, Layers.Recurrent],
					Layers.PerturbateParam:[],
					Layers.Empty:[Layers.FullyConnected, Layers.Convolutional, Layers.Recurrent]
}


activations = {0:'sigmoid', 1:'tanh', 2:'relu', 3:'softmax', 4:'linear'}






