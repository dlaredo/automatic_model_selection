from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import random

"""Rules

	LBlock:1
	FC:2
	Conv:3
	Pooling:4
	RNN:5
	PerturbateParam:6
	Empty:7

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

#Generate a valid subtree
def generate_subtree(parent, last_node=7, next_node=7):

	left_node_type = random.choice(ann_building_rules[last_node])

	right_node_type = random.choice(ann_building_rules[left_node_type])

	#Verification in case some configuration is not valid
	if left_node_type not in ann_building_rules[last_node] or right_node_type not in ann_building_rules[left_node_type]:
		print("Could not generate a valid subtree")
		left_node_type = 7
		right_node_type = 7

	lnode = Node(nodes[left_node_type][0], parent=parent, arity=nodes[left_node_type][1], node_type=left_node_type)
	rnode = Node(nodes[right_node_type][0], parent=parent, arity=nodes[right_node_type][1], node_type=right_node_type)

	last_node = right_node_type

	print(left_node_type)
	print(right_node_type)

	return last_node


def generate_tree():

	tree_model = Node("Model")
	#marc = Node("FC", parent=udo, arity = 2, node_type = 1)
	#lian = Node("Conv", parent=marc, arity = 2, node_type = 1)
	#dan = Node("Recur", parent=udo, arity = 2, node_type = 1)

	last_node = 7

	last_node = generate_subtree(tree_model, last_node)

	while True:

		left = tree_model.children[0]
		right = tree_model.children[1]

		if left.node_type == 1:

		if right.node_type == 1:

		
		print(left)
		print(right)


generate_tree()
#print(udo)
#print(dan)
"""
for pre, fill, node in RenderTree(udo):
	print("%s%s" % (pre, node.name))

# graphviz needs to be installed for the next line!
DotExporter(udo).to_picture("udo.png")
"""







