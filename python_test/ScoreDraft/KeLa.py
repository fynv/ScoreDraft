from .Singer import Singer
from .Catalog import Catalog

try:
	from .PyKeLa import InitializeKeLa, DestroyKeLa
	Catalog['Engines'] += ['KeLa - Singing']

	class KeLa(Singer):
		'''
		Initialize a KeLa based singer.
		path -- path to folder containing the samples.
		'''
		def __init__(self, path):
			self.m_cptr = InitializeKeLa(path)

		def __del__(self):
			DestroyKeLa(self.m_cptr)
			
except ImportError:
	pass




