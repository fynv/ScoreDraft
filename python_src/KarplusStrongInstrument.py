from .Instrument import Instrument
from .Catalog import Catalog

try:
	from .PyKarplusStrongInstrument import InitializeKarplusStrongInstrument, DestroyKarplusStrongInstrument
	from .PyKarplusStrongInstrument import KarplusStrongSetCutFrequency
	from .PyKarplusStrongInstrument import KarplusStrongSetLoopGain
	from .PyKarplusStrongInstrument import KarplusStrongSetSustainGain
	Catalog['Engines'] += ['KarplusStrongInstrument - Instrument']

	class KarplusStrongInstrument(Instrument):
		def __init__(self):
			'''
			Initialize a KarplusStrongInstrument.
			'''
			self.m_cptr = InitializeKarplusStrongInstrument()

		def __del__(self):
			DestroyKarplusStrongInstrument(self.m_cptr)

		def setCutFrequency(self, cut_freq):
			# This is the cut-frequency of the feedback filter for pitch 261.626Hz
			KarplusStrongSetCutFrequency(self.m_cptr, cut_freq)

		def setLoopGain(self, loop_gain):
			KarplusStrongSetLoopGain(self.m_cptr, loop_gain)

		def setSustainGain(self, sustain_gain):
			KarplusStrongSetSustainGain(self.m_cptr, sustain_gain)

except ImportError:
	pass

