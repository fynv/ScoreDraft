from .Singer import Singer
from .Catalog import Catalog

try:
	from .PyUtauDraft import InitializeUtauDraft, DestroyUtauDraft
	from .PyUtauDraft import UtauDraftSetLyricConverter
	from .PyUtauDraft import UtauDraftSetUsePrefixMap
	from .PyUtauDraft import UtauDraftSetCZMode

	Catalog['Engines'] += ['UtauDraft - Singing']

	class UtauDraft(Singer):
		'''
		Initialize a UtauDraft based singer.
		path -- path to the UTAU voicebank.
		'''
		def __init__(self, path, useCUDA=True):
			self.m_cptr = InitializeUtauDraft(path, useCUDA)

		def __del__(self):
			DestroyUtauDraft(self.m_cptr)

		def setLyricConverter(self, lyricConverter):
			'''
			Set a lyric-converter function for a UtauDraft singer used to generate VCV/CVVC lyrics
			The 'LyricConverterFunc' has the following form:

			def LyricConverterFunc(LyricForEachSyllable):
				...
				return [
				(lyric1ForSyllable1, weight11, isVowel11, lyric2ForSyllable1, weight21, isVowel21...  ),
				(lyric1ForSyllable2, weight12, isVowel12, lyric2ForSyllable2, weight22, isVowel22...), ...]

			The argument 'LyricForEachSyllable' has the form [lyric1, lyric2, ...], where each lyric is a string
			In the return value, each lyric is a converted lyric as a string and each weight a float indicating
			the ratio taken within the syllable, plus a bool value indicating whether it is the vowel part of 
			the syllable.
			'''
			UtauDraftSetLyricConverter(self.m_cptr, lyricConverter)

		def setUsePrefixMap(self, usePrefixMap):
			UtauDraftSetUsePrefixMap(self.m_cptr, usePrefixMap)

		def setCZMode(self):
			UtauDraftSetCZMode(self.m_cptr, True)
			
except ImportError:
	pass



