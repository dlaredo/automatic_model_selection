class LNode():

	def __init__(name, node_type, arity):

		self._name = name
		self._node_type = node_type
		self._arity = arity

#Property definition

		@property
		def name(self):
			return self._name

		@name.setter
		def name(self, name):
			self._name = name

		@property
		def node_type(self):
			return self._node_type

		@node_type.setter
		def node_type(self, node_type):
			self._node_type = node_type

		@property
		def arity(self):
			return self.arity

		@arity.setter
		def arity(self, arity):
			self._arity = arity