import os
from .Catalog import Catalog

from .Instrument import Instrument
from .Percussion import Percussion
from .Singer import Singer

from .SimpleInstruments import EnginePureSin
from .SimpleInstruments import EngineSquare
from .SimpleInstruments import EngineTriangle
from .SimpleInstruments import EngineSawtooth
from .SimpleInstruments import EngineNaivePiano
from .SimpleInstruments import EngineBottleBlow
from .SimpleInstruments import PureSin
from .SimpleInstruments import Square
from .SimpleInstruments import Triangle
from .SimpleInstruments import Sawtooth
from .SimpleInstruments import NaivePiano
from .SimpleInstruments import BottleBlow

from .KarplusStrong import EngineKarplusStrong
from .KarplusStrong import KarplusStrongInstrument

from .BasicSamplers import GetSample_Percussion
from .BasicSamplers import GetSample_Single
from .BasicSamplers import GetSamples_Multi
from .BasicSamplers import EnginePercussionSampler
from .BasicSamplers import EngineInstrumentSampler_Single
from .BasicSamplers import EngineInstrumentSampler_Multi
from .BasicSamplers import PercussionSampler
from .BasicSamplers import InstrumentSampler_Single
from .BasicSamplers import InstrumentSampler_Multi

from .SoundFont2 import GetSF2Bank
from .SoundFont2 import ListPresets as ListPresetsSF2
from .SoundFont2 import EngineSoundFont2
from .SoundFont2 import SF2Instrument

from .UtauDraft import GetVoiceBank as GetVoiceBankUTAU
from .UtauDraft import Engine as EngineUtauDraft
from .UtauDraft import UtauDraft

from .CVVCChineseConverter import CVVCChineseConverter
from .XiaYYConverter import XiaYYConverter
from .JPVCVConverter import JPVCVConverter
from .TsuroVCVConverter import TsuroVCVConverter
from .TTEnglishConverter import TTEnglishConverter
from .VCCVEnglishConverter import VCCVEnglishConverter

RESOURCE_ROOT='.'

PERC_SAMPLE_ROOT=RESOURCE_ROOT+'/PercussionSamples'
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

INSTR_SAMPLE_ROOT=RESOURCE_ROOT+'/InstrumentSamples'
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
            definition="""
def """+item+"""():
    return InstrumentSampler_Multi('"""+inst_path+"""')
"""
            exec(definition)
            Catalog['Instruments'] += [item+' - InstrumentSampler_Multi']
            
SF2_ROOT=RESOURCE_ROOT+'/SF2'
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

UTAU_VB_ROOT=RESOURCE_ROOT+'/UTAUVoice'
UTAU_VB_SUFFIX='_UTAU'
if os.path.isdir(UTAU_VB_ROOT):
    for item in os.listdir(UTAU_VB_ROOT):
        if os.path.isdir(UTAU_VB_ROOT+'/'+item):
            definition="""
def """+item+UTAU_VB_SUFFIX+"""(useCuda=True):
    return UtauDraft('"""+UTAU_VB_ROOT+"""/"""+item+"""',useCuda)
"""
            exec(definition)
            Catalog['Singers'] += [item+UTAU_VB_SUFFIX+' - UtauDraft']

