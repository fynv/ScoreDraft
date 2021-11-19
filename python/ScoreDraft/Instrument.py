import numbers

def isNumber(x):
    return isinstance(x, numbers.Number)

def GetTempoMap(tMap, beat48):
    for i in range(1,len(tMap)):
        if beat48 < tMap[i][0] or i == len(tMap)-1:
            return (beat48-tMap[i-1][0])/(tMap[i][0]-tMap[i-1][0])*(tMap[i][1]-tMap[i-1][1])+tMap[i-1][1]
    return 0

class InstrumentShell:
    def __init__(self):
        self.volume=1.0
        self.pan=0.0
    def tune(self,cmd):
        cmd_split= cmd.split(' ')
        cmd_len=len(cmd_split)
        if cmd_len>=1:
            if cmd_len>1 and cmd_split[0]=='volume':
                self.volume=float(cmd_split[1])
                return True
            if cmd_len>1 and cmd_split[0]=='pan':
                self.pan=float(cmd_split[1])
                return True
        return False

    def EnginePlayNote(self, engine, buf, freq, fduration):
        wavBuf=engine.generateWave(freq,fduration, buf.getSampleRate())
        if wavBuf!=None:
            wavBuf.set_volume(self.volume)
            wavBuf.set_pan(self.pan)
            buf.writeBlend(wavBuf)
        buf.moveCursor(fduration)

    def PlayNoteA(self,engine, buf, note, tempoMap, tempoMapOffset, refFreq):
        pos1 = tempoMapOffset
        pos2 = pos1 + note[1]
        fduration =  abs(GetTempoMap(tempoMap, pos2)- GetTempoMap(tempoMap, pos1))
        if note[0]<0.0:
            if note[1]>0.0:
                buf.moveCursor(fduration)
            elif note[1]<0.0:
                buf.moveCursor(-fduration)
            return
        freq = refFreq*note[0]
        self.EnginePlayNote (engine, buf, freq, fduration)

    def PlayNoteB(self, engine, buf, note, tempo, refFreq):
        fduration=abs(note[1]*60000)/(tempo*48)
        if note[0]<0.0:
            if note[1]>0.0:
                buf.moveCursor(fduration)
            elif note[1]<0.0:
                buf.moveCursor(-fduration)
            return
        freq = refFreq*note[0]
        self.EnginePlayNote (engine, buf, freq, fduration)

    def PlaySequence(self, engine, buf, seq, tempo, refFreq):
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
                    tupleSize=len(item)

                    j=0
                    while j<tupleSize:
                        j+=1 # by-pass lyric
                        _item=item[j]
                        if isinstance(_item, (list, tuple)): # singing note
                            while j<tupleSize:
                                _item = item[j]
                                if not isinstance(_item, (list, tuple)):
                                    break
                                note=(_item[0],_item[1])

                                if using_tempo_map:
                                    self.PlayNoteA(engine,buf, note, tempo_map, beatPos, refFreq)
                                else:
                                    self.PlayNoteB(engine,buf, note, tempo, refFreq)
                                beatPos+=note[1]
                                j+=1

                        elif isNumber(_item): # singing rap
                            duration = item[j]
                            note=(item[j+1], duration)
                            if using_tempo_map:
                                self.PlayNoteA(engine,buf, note, tempo_map, beatPos, refFreq)
                            else:
                                self.PlayNoteB(engine,buf, note, tempo, refFreq)
                            beatPos+=note[1]
                            j+=3

                elif isNumber(_item): # note
                    note = (item[0],item[1])
                    if using_tempo_map:
                        self.PlayNoteA(engine,buf, note, tempo_map, beatPos, refFreq)
                    else:
                        self.PlayNoteB(engine,buf, note, tempo, refFreq)
                    beatPos+=note[1]
            elif type(item)== str:
                if not self.tune(engine,item):
                    engine.tune(item)


class Instrument:
    def __init__(self):
        self.shell=InstrumentShell()
    def play(self, buf, seq, tempo=80.0, refFreq=261.626):
        self.shell.PlaySequence(self.engine, buf, seq, tempo, refFreq)
    def tune(self,cmd):
        if not self.shell.tune(cmd):
            self.engine.tune(cmd)
    def setNoteVolume(self,volume):
        self.shell.volume=volume
    def setNotePan(self,pan):
        self.shell.pan=pan
    def isGMDrum(self):
        return False
