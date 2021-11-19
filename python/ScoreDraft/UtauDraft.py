import wave
from .ScoreDraftCore import F32Buf
from .UTAUUtils import VoiceBank
from .Singer import Singer
from .Catalog import Catalog
Catalog['Engines'] += ['UtauDraft - Singing']

# VoiceSampler
notVowel = 0
preVowel = 1
isVowel = 2
from .VoiceSampler import GenerateSentence
from .VoiceSampler import GenerateSentenceCUDA

VoiceBanks={}

def loadWav(file):
    wavS16=bytes()
    with wave.open(file, mode='rb') as wavFile:
        wavS16=wavFile.readframes(wavFile.getnframes())
    return F32Buf.from_s16(wavS16)
    
def GetVoiceBank(path):
    if not (path in VoiceBanks):
        VoiceBanks[path ]= VoiceBank(path)
    return VoiceBanks[path]
    
def _PieceMapper(syllableLyricList, lyricList, otoList, CZMode=False):
    transition = 0.1

    result= {
        'maps': [], 
        'piece_map': []
    }
    maps= result['maps']
    map1=[(otoList[0][1], otoList[0][1]- otoList[0][2], notVowel)]
    maps+=[map1]
    map0 = map1
    map1 = []

    piece_map = result['piece_map']
    piece_map += [(0,0)]

    cursor=0
    for i in range(len(lyricList)):
        piece = lyricList[i]
        piece_oto = otoList[i]
        piece_oto_next = None
        if i<len(lyricList)-1:
            piece_oto_next =  otoList[i+1]

        piece_isVowel = piece['isVowel']
        piece_duration = piece['weight']*syllableLyricList[piece['syllableId']]['duration']

        if (not CZMode) or piece_isVowel:
            srcTotalLen= piece_oto[4] -  piece_oto[2];
            if piece_oto_next != None:
                srcTotalLen += piece_oto_next[2] - piece_oto_next[1]

            srcVowelLen=0
            srcFixedLen = srcTotalLen

            if piece_isVowel:
                srcVowelLen = piece_oto[4] -  piece_oto[3]
                srcFixedLen -= srcVowelLen

            vowelScale=1.0
            fixedScale=1.0

            if srcVowelLen>0 and srcTotalLen<piece_duration:
                vowelScale = (piece_duration-srcFixedLen)/srcVowelLen
            else:
                fixedScale = piece_duration/srcTotalLen
                vowelScale = fixedScale

            curLen = fixedScale * (piece_oto[3] -  piece_oto[2]) + vowelScale * (piece_oto[4] -  piece_oto[3])
            if piece_oto_next != None:
                piece_map += [(i, cursor+curLen*(1.0-transition)), (i+1, cursor+curLen)]
            else:
                piece_map += [(i, cursor+curLen)]


            seg_isVowel = preVowel
            if not piece_isVowel:
                seg_isVowel = notVowel
            map0+=[(piece_oto[2], cursor, seg_isVowel)]

            cursor += fixedScale * (piece_oto[3] -  piece_oto[2])
            seg_isVowel = isVowel
            if not piece_isVowel:
                seg_isVowel  = notVowel
            map0+=[(piece_oto[3], cursor, seg_isVowel)]

            cursor += vowelScale * (piece_oto[4] -  piece_oto[3])
            map0+=[(piece_oto[4], cursor)]

            if piece_oto_next != None:
                map1+=[(piece_oto_next[0], cursor +  vowelScale *(piece_oto_next[0]-piece_oto_next[1]), seg_isVowel)]
                map1+=[(piece_oto_next[1], cursor, notVowel)]
                cursor += fixedScale * (piece_oto_next[2] - piece_oto_next[1])
                maps+=[map1]

        else:  # CZ Mode
            srcTotalLen = 80.0
            srcCurLen = 80.0
            if piece_oto_next != None:
                srcTotalLen = piece_oto_next[2] - piece_oto_next[0]
                srcCurLen = piece_oto_next[1] - piece_oto_next[0]
            scale =  piece_duration/srcTotalLen
            curLen = piece_duration
            if piece_oto_next != None:
                curLen = scale*srcCurLen
                piece_map += [(i, cursor+curLen*(1.0-transition)), (i+1, cursor+curLen)]
            else:
                piece_map += [(i, cursor+curLen)]

            map0+=[(piece_oto[2], cursor, notVowel)]
            cursor += curLen
            map0+=[(piece_oto[2] + srcCurLen, cursor)]

            if piece_oto_next != None:
                map1+=[(piece_oto_next[0], cursor +  scale *(piece_oto_next[0]-piece_oto_next[1]), notVowel)]
                map1+=[(piece_oto_next[1], cursor, notVowel)]
                cursor += scale * (piece_oto_next[2] - piece_oto_next[1])
                maps+=[map1]

        map0 = map1
        map1 = []

    return result

def DefaultPieceMapper(syllableLyricList, lyricList, otoList):
    return _PieceMapper(syllableLyricList, lyricList, otoList, False)

def CZPieceMapper(syllableLyricList, lyricList, otoList):
    return _PieceMapper(syllableLyricList, lyricList, otoList, True)
    
class Engine:
    def __init__(self, voiceBank):
        if type(voiceBank)==str:
            voiceBank=GetVoiceBank(voiceBank)
        if not voiceBank.initialized:
            voiceBank.initialize()
        self.voiceBank=voiceBank
        self.lyricConverter=None
        self.usePrefixMap = True
        self.pieceMapper = DefaultPieceMapper
        self.useCUDA = True

    def tune(self, cmd):
        cmd_split= cmd.split(' ')
        cmd_len=len(cmd_split)
        if cmd_len>=1:
            if cmd_len>1 and cmd_split[0]=='prefix_map':
                if cmd_split[1]=='on':
                    self.usePrefixMap = True
                    return True
                elif cmd_split[1]=='off':
                    self.usePrefixMap = False
                    return True
        return False

    def _convertLyric(self, syllableList):
        convertedList = self.lyricConverter([syllable['lyric'] for syllable in syllableList])
        lyricList=[]
        for i in range(len(convertedList)):
            convertedSyllable= convertedList[i]
            sumWeight=0
            for j in range(len(convertedSyllable)//3):
                sumWeight += convertedSyllable[j*3+1]

            for j in range(len(convertedSyllable)//3):
                piece={
                    'lyric': convertedSyllable[j*3],
                    'weight': convertedSyllable[j*3+1]/sumWeight,
                    'isVowel': convertedSyllable[j*3+2],
                    'syllableId': i
                }
                lyricList+=[piece]
        return lyricList

    def generateWave(self, syllableList, sampleRate):
        # print(syllableList)
        syllableLyricList=[]
        totalDuration=0
        for syllable in syllableList:
            aveFreq=0
            duration=0
            for j in range(len(syllable['ctrlPnts'])):
                ctrlPnt=syllable['ctrlPnts'][j]
                if ctrlPnt[1] <=0:
                    continue
                freq1 = ctrlPnt[0]
                freq2 = freq1
                if j < len(syllable['ctrlPnts']) -1:
                    freq2 = syllable['ctrlPnts'][j+1][0]
                aveFreq +=(freq1 + freq2) * ctrlPnt[1]
                duration+=ctrlPnt[1]
            aveFreq *= 1.0/duration*0.5;
            syllablePiece={
                'lyric': syllable['lyric'],
                'duration' : duration,
                'aveFreq': aveFreq
            }
            syllableLyricList+=[syllablePiece]
            totalDuration+=duration

        lyricList=[]
        if self.lyricConverter == None:
            for i in range(len(syllableLyricList)):
                syllablePiece=syllableLyricList[i]
                piece={
                    'lyric': syllablePiece['lyric'],
                    'weight': 1.0,
                    'isVowel': True,
                    'syllableId': i
                }
                lyricList+=[piece]
        else:
            lyricList=self._convertLyric(syllableLyricList)

        srcList = []
        otoList = []
        for piece in lyricList:
            wavFrq= self.voiceBank.getWavFrq_PrefixMap(piece['lyric'], syllableLyricList[piece['syllableId']]['aveFreq'])
            wavFileName=wavFrq[0]['filename']
            wav = loadWav(wavFileName)
            srcList += [
                {
                    'wav': wav,
                    'frq': wavFrq[1]
                }
            ]
            start = wavFrq[0]['offset']
            overlap  = start + wavFrq[0]['overlap']
            preutterance = start + wavFrq[0]['preutterance']
            consonant = start + wavFrq[0]['consonant']
            oto_cutoff= wavFrq[0]['cutoff']
            end = 0
            if oto_cutoff >=0:
                end = wav.size()/44.1 - oto_cutoff
            else:
                end = start - oto_cutoff
            otoList += [(start, overlap, preutterance, consonant, end)]

        piece_map = self.pieceMapper(syllableLyricList, lyricList, otoList)

        sentence= {
            'pieces': [],
            'piece_map': [],
            'freq_map': [],
            'volume_map': []
        }

        for i in range(len(lyricList)):
            sentence['pieces']+=[
                {
                    'src': srcList[i],
                    'map': piece_map['maps'][i]
                }
            ]
        sentence['piece_map'] = piece_map['piece_map']
        freq_map= sentence['freq_map']
        cursor=0
        for syllable in syllableList:
            for ctrlPnt in syllable['ctrlPnts']:
                freq_map+=[(ctrlPnt[0], cursor)]
                cursor+=ctrlPnt[1]
        volume_map = sentence['volume_map']
        volume_map += [(1.0, 0)]
        last_duration = syllableLyricList[len(syllableLyricList)-1]['duration']
        volume_map += [(1.0, totalDuration-last_duration*0.1)]
        volume_map += [(0.0, totalDuration)]

        if self.useCUDA:
            return GenerateSentenceCUDA(sentence)
        else:
            return GenerateSentence(sentence)

class UtauDraft(Singer):
    def __init__(self, voiceBank, useCUDA=True):
        Singer.__init__(self)
        self.engine=Engine(voiceBank)
        self.engine.useCUDA=useCUDA
    def setLyricConverter(self, lyricConverter):
        self.engine.lyricConverter=lyricConverter
    def setPieceMapper(self, pieceMapper):
        self.engine.pieceMapper=pieceMapper
    def setUsePrefixMap(self,usePrefixMap):
        self.engine.usePrefixMap=usePrefixMap
    def setCZMode(self):
        self.engine.pieceMapper= CZPieceMapper


