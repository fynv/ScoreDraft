from .Instrument import Instrument
from .Catalog import Catalog

try:
	from .Extensions import InitializeKarplusStrongInstrument
	from .Extensions import KarplusStrongSetCutFrequency
	from .Extensions import KarplusStrongSetLoopGain
	from .Extensions import KarplusStrongSetSustainGain
	Catalog['Engines'] += ['KarplusStrongInstrument - Instrument']

	class KarplusStrongInstrument(Instrument):
		def __init__(self):
			self.id = InitializeKarplusStrongInstrument()
		def setCutFrequency(self, cut_freq):
			# This is the cut-frequency of the feedback filter for pitch 261.626Hz
			KarplusStrongSetCutFrequency(self, cut_freq)
		def setLoopGain(self, loop_gain):
			KarplusStrongSetLoopGain(self, loop_gain)
		def setSustainGain(self, sustain_gain):
			KarplusStrongSetSustainGain(self, sustain_gain)

except ImportError:
	pass

