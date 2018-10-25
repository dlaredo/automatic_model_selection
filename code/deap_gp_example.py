from deap import base, creator, gp
import operator

pset = gp.PrimitiveSet("main", 3)
pset.addPrimitive(max, 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addTerminal(3)
pset.addPrimitive(operator.neg, 1)

pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")
pset.renameArguments(ARG2="z")

expr = gp.genFull(pset, min_=1, max_=3)
tree = gp.PrimitiveTree(expr)

print(tree)