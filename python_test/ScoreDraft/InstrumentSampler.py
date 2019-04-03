from .Instrument import Instrument
from .Catalog import Catalog

try:
	from .PyInstrumentSampler import InitializeInstrumentSingleSampler, InitializeInstrumentMultiSampler, DestroyInstrumentSampler
	Catalog['Engines'] += ['InstrumentSampler_Single - Instrument']
	Catalog['Engines'] += ['InstrumentSampler_Multi - Instrument']

	class InstrumentSampler_Single(Instrument):
		'''
		Initialize a instrument sampler using a single .wav file.
		wavPath -- path to the .wav file.
		'''
		def __init__(self, wavPath):
			self.m_cptr = InitializeInstrumentSingleSampler(wavPath)

		def __del__(self):
			DestroyInstrumentSampler(self.m_cptr)

	class InstrumentSampler_Multi(Instrument):
		'''
		Initialize a instrument sampler using multiple .wav files.
		folderPath -- path containining the .wav files
		'''
		def __init__(self, folderPath):
			self.m_cptr = InitializeInstrumentMultiSampler(folderPath)

		def __del__(self):
			DestroyInstrumentSampler(self.m_cptr)
			
except ImportError:
	pass
	