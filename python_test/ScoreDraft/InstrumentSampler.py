from .Instrument import Instrument
from .Catalog import Catalog

try:
	from .Extensions import InitializeInstrumentSingleSampler
	Catalog['Engines'] += ['InstrumentSampler_Single - Instrument']

	class InstrumentSampler_Single(Instrument):
		def __init__(self, wavPath):
			self.id = InitializeInstrumentSingleSampler(wavPath)
except ImportError:
	pass


try:
	from .Extensions import InitializeInstrumentMultiSampler
	Catalog['Engines'] += ['InstrumentSampler_Multi - Instrument']

	class InstrumentSampler_Multi(Instrument):
		def __init__(self, folderPath):
			self.id = InitializeInstrumentMultiSampler(folderPath)
except ImportError:
	pass
	