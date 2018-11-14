import os
from . import PyScoreDraft

ScoreDraftPath_old= os.path.dirname(__file__)
ScoreDraftPath=""
#\\escaping fix
for ch in ScoreDraftPath_old:
	if ch=="\\":
		ScoreDraftPath+="/"
	else:
		ScoreDraftPath+=ch

if os.name == 'nt':
	os.environ["PATH"]+=";"+ScoreDraftPath
elif os.name == "posix":
	os.environ["PATH"]+=":"+ScoreDraftPath

PyScoreDraft.ScanExtensions(ScoreDraftPath)

from .PyScoreDraft import TellDuration
'''
TellDuration(seq) takes in a single input "seq"
It can be a note-sequence, a beat-sequence, or a singing-sequence, 
anything acceptable by Instrument.play(), Percussion.play(), Singer.sing()
as the "seq" parameter
The return value is the total duration of the sequence as an integer
'''

from .TrackBuffer import setDefaultNumberOfChannels
from .TrackBuffer import TrackBuffer
from .TrackBuffer import MixTrackBufferList
from .TrackBuffer import WriteTrackBufferToWav
from .TrackBuffer import ReadTrackBufferFromWav

try:
	from .Extensions import WriteNoteSequencesToMidi
except ImportError:
	pass

try:
	from .Extensions import PlayTrackBuffer
except ImportError:
	pass

try:
	from .Extensions import PlayGetRemainingTime
except ImportError:
	pass

try:
	from .Extensions import QPlayTrackBuffer
except ImportError:
	pass

try:
	from .Extensions import QPlayGetRemainingTime
except ImportError:
	pass

from .Catalog import Catalog
from .Catalog import PrintCatalog

from .Instrument import Instrument
from .Percussion import Percussion
from .Singer import Singer
from .Document import Document

try:
	from .Meteor import Document as MeteorDocument
except ImportError:
	pass

from .InternalInstruments import PureSin, Square, Triangle, Sawtooth, NaivePiano, BottleBlow

try:
	from .PercussionSampler import PercussionSampler

	PERC_SAMPLE_ROOT=ScoreDraftPath+'/PercussionSamples'
	if os.path.isdir(PERC_SAMPLE_ROOT):
		for item in os.listdir(PERC_SAMPLE_ROOT):
			file_path = PERC_SAMPLE_ROOT+'/'+item
			if os.path.isfile(file_path) and item.endswith(".wav"):
				name = item[0:len(item)-4]
				definition="""
def """+name+"""():
	return PercussionSampler('"""+file_path+"""')
"""
				exec(definition)
				Catalog['Percussions'] += [name+' - PercussionSampler']
except ImportError:
	pass

try:
	from .InstrumentSampler import InstrumentSampler_Single
	from .InstrumentSampler import InstrumentSampler_Multi

	INSTR_SAMPLE_ROOT=ScoreDraftPath+'/InstrumentSamples'
	if os.path.isdir(INSTR_SAMPLE_ROOT):
		for item in os.listdir(INSTR_SAMPLE_ROOT):
			inst_path = INSTR_SAMPLE_ROOT+'/'+item
			if os.path.isfile(inst_path) and item.endswith(".wav"):
				name = item[0:len(item)-4]
				definition="""
def """+name+"""():
	return InstrumentSampler_Single('"""+inst_path+"""')
"""
				exec(definition)
				Catalog['Instruments'] += [name+' - InstrumentSampler_Single']
			elif os.path.isdir(inst_path):
				name = item
				definition="""
def """+item+"""():
	return InstrumentSampler_Multi('"""+inst_path+"""')
"""
				exec(definition)
				Catalog['Instruments'] += [name+' - InstrumentSampler_Multi']
except ImportError:
	pass

try:
	from .KeLa import KeLa

	KELA_SAMPLE_ROOT=ScoreDraftPath+'/KeLaSamples'
	if os.path.isdir(KELA_SAMPLE_ROOT):
		for item in os.listdir(KELA_SAMPLE_ROOT):
			kela_path = KELA_SAMPLE_ROOT+'/'+item
			if os.path.isdir(kela_path):
				definition="""
def """+item+"""():
	return KeLa('"""+kela_path+"""')
"""
				exec(definition)
				Catalog['Singers'] += [item+' - KeLa']
except ImportError:
	pass

try:
	from .UtauDraft import UtauDraft
	from .CVVCChineseConverter import CVVCChineseConverter
	from .XiaYYConverter import XiaYYConverter
	from .JPVCVConverter import JPVCVConverter
	from .TsuroVCVConverter import TsuroVCVConverter
	from .TTEnglishConverter import TTEnglishConverter
	from .VCCVEnglishConverter import VCCVEnglishConverter

	UTAU_VB_ROOT=ScoreDraftPath+'/UTAUVoice'
	UTAU_VB_SUFFIX='_UTAU'
	if os.path.isdir(UTAU_VB_ROOT):
		for item in os.listdir(UTAU_VB_ROOT):
			utau_path = UTAU_VB_ROOT+'/'+item
			if os.path.isdir(utau_path):
				definition="""
def """+item+UTAU_VB_SUFFIX+"""(useCuda=True):
	return UtauDraft('"""+utau_path+"""',useCuda)
"""
				exec(definition)
				Catalog['Singers'] += [item+UTAU_VB_SUFFIX+' - UtauDraft']
except ImportError:
	pass


try:
	from .SF2Instrument import ListPresets as ListPresetsSF2
	from .SF2Instrument import SF2Instrument

	SF2_ROOT=ScoreDraftPath+'/SF2'
	if os.path.isdir(SF2_ROOT):
		for item in os.listdir(SF2_ROOT):
			sf2_path = SF2_ROOT+'/'+item
			if os.path.isfile(sf2_path) and item.endswith(".sf2"):
				name = item[0:len(item)-4]
				definition="""
def """+name+"""(preset_index):
	return SF2Instrument('"""+sf2_path+"""', preset_index)

def """+name+"""_List():
	ListPresetsSF2('"""+sf2_path+"""')
"""
				exec(definition)
				Catalog['Instruments'] += [name+' - SF2Instrument']

except ImportError:
	pass	

try:
	from .KarplusStrongInstrument import KarplusStrongInstrument
except ImportError:
	pass	
