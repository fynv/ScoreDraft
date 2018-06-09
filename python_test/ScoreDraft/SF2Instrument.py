from .Instrument import Instrument
from .Catalog import Catalog

try:
	from .Extensions import InitializeSF2Instrument
	from .Extensions import SF2InstrumentListPresets as ListPresets
	Catalog['Engines'] += ['SF2Instrument - Instrument']

	class SF2Instrument(Instrument):
		def __init__(self, sfPath, preset_index):
			self.id = InitializeSF2Instrument(sfPath, preset_index)
except ImportError:
	pass


