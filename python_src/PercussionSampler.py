from .Percussion import Percussion
from .Catalog import Catalog

try:
	from .PyPercussionSampler import InitializePercurssionSampler, DestroyPercurssionSampler

	Catalog['Engines'] += ['PercussionSampler - Percussion']

	class PercussionSampler(Percussion):
		'''
		Initialize a percussion sampler using a single .wav file.
		wavPath -- path to the .wav file.
		'''
		def __init__(self, wavPath):
			self.m_cptr = InitializePercurssionSampler(wavPath)

		def __del__(self):
			DestroyPercurssionSampler(self.m_cptr)

except ImportError:
	pass




