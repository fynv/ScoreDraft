import numbers

def isNumber(x):
    return isinstance(x, numbers.Number)

def TellDuration(seq):
    duration = 0
    for item in seq:
        if isinstance(item, (list, tuple)):
            _item = item[0] 
            if type(_item) == str: # singing
                tupleSize=len(item)
                j=0
                while j<tupleSize:
                    j+=1  # by-pass lyric
                    _item=item[j]
                    if isinstance(_item, (list, tuple)): # singing note
                        while j<tupleSize:
                            _item = item[j]
                            if not isinstance(_item, (list, tuple)):
                                break
                            numCtrlPnt= len(_item)//2
                            for k in range(numCtrlPnt):
                                duration+=_item[k*2+1]
                            j+=1
                    elif isNumber(_item): # singing rap
                        duration += item[j]
                        j+=3
            elif isNumber(_item): # note
                duration += item[1]
    return duration



from .ScoreDraftCore import F32Buf
from .ScoreDraftCore import WavBuffer
from .ScoreDraftCore import setDefaultNumberOfChannels
from .ScoreDraftCore import TrackBuffer
from .ScoreDraftCore import MixTrackBufferList
from .ScoreDraftCore import WriteTrackBufferToWav
from .ScoreDraftCore import ReadTrackBufferFromWav

from .UTAUUtils import LoadFrq as LoadFrqUTAU
from .UTAUUtils import LoadOtoINIPath as LoadOtoINIPathUTAU
from .UTAUUtils import LoadPrefixMap as LoadPrefixMapUTAU
from .UTAUUtils import LookUpPrefixMap as LookUpPrefixMapUTAU
from .UTAUUtils import VoiceBank as VoiceBankUTAU

notVowel = 0
preVowel = 1
isVowel = 2

from .VoiceSampler import GenerateSentence
from .VoiceSampler import GenerateSentenceCUDA
from . import VoiceSampler

def DetectFrqVoice(wavF32, interval=256):
    frq_data = VoiceSampler.FrqData()
    frq_data.detect(wavF32, interval)
    return frq_data
    
from .Catalog import PrintCatalog
from .Initializers import *

from .Document import Document
try:
    from .Meteor import Meteor, MeteorPlay
    from .Meteor import Document as MeteorDocument
except:
    print('Meteor import failed')

from .MIDIWriter import WriteNoteSequencesToMidi
try:
    from .PCMPlayer import PCMPlayer, AsyncUIPCMPlayer
except:
    print('PCMPlayer import failed')    

def PlayTrackBuffer(track):
    player = AsyncUIPCMPlayer()
    player.play_track(track)

try:
    from .MusicXMLDocument import MusicXMLDocument, from_music_xml, from_lilypond
except:
    print('MusicXMLDocument import failed')

try:
    from .YAMLDocument import YAMLScore, YAMLDocument
    has_yaml = True
except:
    print('YAMLDocument import failed')
    has_yaml = False
    
import argparse
    
def run_yaml():
    if has_yaml:
        parser = argparse.ArgumentParser(prog = "scoredraft")
        parser.add_argument("yaml", help = "input yaml filename")
        parser.add_argument("-ly", help = "output lilyond filename")
        parser.add_argument("-wav", help = "output wav filename")
        parser.add_argument("-meteor", help = "output meteor filename")
        parser.add_argument("-run", help = "run meteor", action='store_true')
        args=vars(parser.parse_args())
        with open(args['yaml'], 'r', encoding = 'utf-8') as f_in:
            score = YAMLScore(f_in)
            if not args['ly'] is None:
                with open(args['ly'], 'w', encoding = 'utf-8') as f_out:
                    f_out.write(score.to_ly())
            if not args['wav'] is None or not args['meteor'] is None or args['run']:
                doc = YAMLDocument(score)
                doc.play()
                if not args['wav'] is None:
                    doc.mixDown(args['wav'])
                if not args['meteor'] is None:
                    doc.saveToFile(args['meteor'])
                if args['run']:
                    doc.meteor()

