from .Singer import Singer
from .Catalog import Catalog

try:
	from .Extensions import InitializeKeLa

	Catalog['Engines'] += ['KeLa - Singing']

	class KeLa(Singer):
		def __init__(self, path):
			self.id = InitializeKeLa(path)
except ImportError:
	pass




