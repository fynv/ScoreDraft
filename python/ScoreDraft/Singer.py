import numbers

def isNumber(x):
    return isinstance(x, numbers.Number)

def GetTempoMap(tMap, beat48):
    for i in range(1,len(tMap)):
        if beat48 < tMap[i][0] or i == len(tMap)-1:
            return (beat48-tMap[i-1][0])/(tMap[i][0]-tMap[i-1][0])*(tMap[i][1]-tMap[i-1][1])+tMap[i-1][1]
    return 0

class SingerShell:
    def __init__(self):
        self.default_lyric='a'
        self.volume=1.0
        self.pan=0.0
    def tune(self,cmd):
        cmd_split= cmd.split(' ')
        cmd_len=len(cmd_split)
        if cmd_len>=1:
            if cmd_len>1 and cmd_split[0]=='default_lyric':
                self.default_lyric=cmd_split[1]
                return True
            if cmd_len>1 and cmd_split[0]=='volume':
                self.volume=float(cmd_split[1])
                return True
            if cmd_len>1 and cmd_split[0]=='pan':
                self.pan=float(cmd_split[1])
                return True
        return False

    def EngineSingSyllables(self,engine, buf, syllableList,totalDuration):
        wavBuf=engine.generateWave(syllableList, buf.getSampleRate())
        if wavBuf!=None:
            wavBuf.set_volume(self.volume)
            wavBuf.set_pan(self.pan)
            buf.writeBlend(wavBuf)
        buf.moveCursor(totalDuration)

    def SingSyllablesA(self,engine, buf, syllables, tempoMap, tempoMapOffset, refFreq):
        syllableList=[]
        totalDuration = 0
        beatPos = tempoMapOffset
        for syllable in syllables:
            ctrlPnts=[]
            for aCtrlPnt in syllable['ctrlPnts']:
                pos1=beatPos
                pos2 = pos1 + aCtrlPnt[1]
                fduration = abs(GetTempoMap(tempoMap, pos2)- GetTempoMap(tempoMap, pos1))
                if aCtrlPnt[0]<0:
                    if len(syllableList)>0 or len(ctrlPnts)>0:
                        if len(ctrlPnts)>0:
                            _syllable = {
                                'lyric': syllable['lyric'],
                                'ctrlPnts' : ctrlPnts
                            }
                            syllableList+=[_syllable]
                        self.EngineSingSyllables(engine, buf, syllableList, totalDuration)
                        ctrlPnts=[]
                        syllableList=[]
                        totalDuration = 0
                    if aCtrlPnt[1]>0:
                        buf.moveCursor(fduration)
                    elif aCtrlPnt[1]<0:
                        buf.moveCursor(-fduration)
                    continue
                freq= refFreq*aCtrlPnt[0]
                ctrlPnts+=[(freq, fduration)]
                totalDuration+=fduration
                beatPos = pos2;
            if len(ctrlPnts)>0:
                _syllable = {
                    'lyric': syllable['lyric'],
                    'ctrlPnts' : ctrlPnts
                }
                syllableList+=[_syllable]
        if len(syllableList)>0:
            self.EngineSingSyllables(engine, buf, syllableList, totalDuration)  


    def SingSyllablesB(self, engine, buf, syllables, tempo, refFreq):
        syllableList=[]
        totalDuration = 0

        for syllable in syllables:
            ctrlPnts=[]
            for aCtrlPnt in syllable['ctrlPnts']:
                fduration=abs(aCtrlPnt[1]*60000)/(tempo*48)
                if aCtrlPnt[0]<0:
                    if len(syllableList) or len(ctrlPnts)>0:
                        if len(ctrlPnts)>0:
                            _syllable = {
                                'lyric': syllable['lyric'],
                                'ctrlPnts' : ctrlPnts
                            }
                            syllableList+=[_syllable]
                        self.EngineSingSyllables(engine, buf, syllableList, totalDuration)
                        ctrlPnts=[]
                        syllableList=[]
                        totalDuration = 0
                    if aCtrlPnt[1]>0:
                        buf.moveCursor(fduration)
                    elif aCtrlPnt[1]<0:
                        buf.moveCursor(-fduration)
                    continue
                freq= refFreq*aCtrlPnt[0]
                ctrlPnts+=[(freq, fduration)]
                totalDuration+=fduration
            if len(ctrlPnts)>0:
                _syllable = {
                    'lyric': syllable['lyric'],
                    'ctrlPnts' : ctrlPnts
                }
                syllableList+=[_syllable]
        if len(syllableList)>0:
            self.EngineSingSyllables(engine, buf, syllableList, totalDuration)


    def SingSequence(self,engine, buf, seq, tempo, refFreq):
        using_tempo_map= (type(tempo)== list)
        tempo_map=[]
        if using_tempo_map:
            
            cursor = buf.getCursor()
            if tempo[0][0] == 0:
                cursor= tempo[0][1]
                buf.setCursor(cursor)
            else:
                ctrlPnt=(0, cursor)
                tempo_map+=[ctrlPnt]

            for ctrlPnt in tempo:
                tempo_map+=[ctrlPnt]
                
        beatPos=0
        for item in seq:
            if isinstance(item, (list, tuple)):
                _item = item[0] 
                if type(_item) == str: # singing
                    totalDuration = 0
                    syllables=[]
                    tupleSize=len(item)

                    j=0
                    while j<tupleSize:
                        lyric=item[j]
                        if len(lyric)==0:
                            lyric=self.default_lyric
                        j+=1
                        _item=item[j]
                        if isinstance(_item, (list, tuple)): # singing note
                            syllable={'lyric': lyric, 'ctrlPnts':[]}
                            while j<tupleSize:
                                _item = item[j]
                                if not isinstance(_item, (list, tuple)):
                                    break
                                numCtrlPnt= (len(_item)+1)//2
                                for k in range(numCtrlPnt):
                                    freq_rel=_item[k*2]
                                    duration= 0
                                    if k*2+1<len(_item):
                                        duration=_item[k*2+1]
                                        totalDuration += duration
                                    ctrlPnt=(freq_rel,duration)
                                    syllable['ctrlPnts']+=[ctrlPnt]

                                lastCtrlPnt = syllable['ctrlPnts'][len(syllable['ctrlPnts'])-1]
                                if lastCtrlPnt[0] > 0 and lastCtrlPnt[1]>0:
                                    ctrlPnt=(lastCtrlPnt[0], 0)
                                    syllable['ctrlPnts']+=[ctrlPnt]
                                j+=1
                            syllables+=[syllable]
                        elif isNumber(_item): # singing rap
                            syllable={'lyric': lyric, 'ctrlPnts':[]}
                            duration = item[j]
                            j+=1
                            freq1 = item[j]
                            j+=1
                            freq2 = item[j]
                            j+=1

                            if freq1 > 0 and freq2>0:
                                syllable['ctrlPnts']+=[(freq1, duration), (freq2, 0)]
                            else:
                                syllable['ctrlPnts']+=[(-1, duration)]
                            totalDuration += duration
                            syllables+=[syllable]

                    if len(syllables)>0:
                        if using_tempo_map:
                            self.SingSyllablesA(engine,buf, syllables, tempo_map, beatPos, refFreq)
                        else:
                            self.SingSyllablesB(engine,buf, syllables, tempo, refFreq)
                    
                    beatPos+=totalDuration

                elif isNumber(_item): # note
                    syllable={'lyric': self.default_lyric, 'ctrlPnts':[(item[0],item[1])]}
                    if item[0]>0:
                        syllable['ctrlPnts']+=[(item[0],0)]
                    if using_tempo_map:
                        self.SingSyllablesA(engine,buf, [syllable], tempo_map, beatPos, refFreq)
                    else:
                        self.SingSyllablesB(engine,buf, [syllable], tempo, refFreq)
                    beatPos+=item[1]
            elif type(item)== str:
                if not self.tune(engine,item):
                    engine.tune(item)


class Singer:
    def __init__(self):
        self.shell=SingerShell()
    def sing(self, buf, seq, tempo=80, refFreq=261.626):
        self.shell.SingSequence(self.engine, buf, seq, tempo,refFreq)
    def tune(self,cmd):
        if not self.shell.tune(cmd):
            self.engine.tune(cmd)
    def setDefaultLyric(self,defaultLyric):
        self.shell.default_lyric=defaultLyric
    def setNoteVolume(self,volume):
        self.shell.volume=volume
    def setNotePan(self,pan):
        self.shell.pan=pan

