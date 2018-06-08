from .Percussion import Percussion
from .Catalog import Catalog

try:
	from .Extensions import InitializePercurssionSampler

	Catalog['Engines'] += ['PercussionSampler - Percussion']

	class PercussionSampler(Percussion):
		def __init__(self, wavPath):
			self.id = InitializePercurssionSampler(wavPath)
except ImportError:
	pass




