from .Singer import Singer
from .Catalog import Catalog

try:
	from .Extensions import InitializeUtauDraft
	from .Extensions import UtauDraftSetLyricConverter
	from .Extensions import UtauDraftSetUsePrefixMap
	from .Extensions import UtauDraftSetCZMode


	Catalog['Engines'] += ['UtauDraft - Singing']

	class UtauDraft(Singer):
		def __init__(self, path, useCUDA=True):
			self.id = InitializeUtauDraft(path, useCUDA)

		def setLyricConverter(self, lyricConverter):
			UtauDraftSetLyricConverter(self, lyricConverter)

		def setUsePrefixMap(self, usePrefixMap):
			UtauDraftSetUsePrefixMap(self, usePrefixMap)

		def setCZMode(self):
			UtauDraftSetCZMode(self, True)
except ImportError:
	pass



