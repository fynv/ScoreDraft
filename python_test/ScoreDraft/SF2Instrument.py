from .Instrument import Instrument
from .Catalog import Catalog

try:
	from .PySF2Instrument import InitializeSF2Instrument, DestroySF2Instrument
	from .PySF2Instrument import SF2InstrumentListPresets as ListPresets
	'''
	List presets of a sf2 file
	sf2Path -- path to the sf2 file.
	'''
	Catalog['Engines'] += ['SF2Instrument - Instrument']

	class SF2Instrument(Instrument):
		'''
		Initialize a SF2 based instrument
		sf2Path -- path to the sf2 file.
		preset_index -- preset index.
		'''
		def __init__(self, sfPath, preset_index):
			self.m_cptr = InitializeSF2Instrument(sfPath, preset_index)

		def __del__(self):
			DestroySF2Instrument(self.m_cptr)

except ImportError:
	pass


