import numbers

def isNumber(x):
    return isinstance(x, numbers.Number)

def GetTempoMap(tMap, beat48):
    for i in range(1,len(tMap)):
        if beat48 < tMap[i][0] or i == len(tMap)-1:
            return (beat48-tMap[i-1][0])/(tMap[i][0]-tMap[i-1][0])*(tMap[i][1]-tMap[i-1][1])+tMap[i-1][1]
    return 0

class PercussionShell:
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

    def EnginePlayBeat(self, engine, buf, fduration):
        wavBuf=engine.generateWave(fduration, buf.getSampleRate())
        if wavBuf!=None:
            wavBuf.set_volume(self.volume)
            wavBuf.set_pan(self.pan)
            buf.writeBlend(wavBuf)
        buf.moveCursor(fduration)

    def PlayBeatA(self, engine, buf, duration, tempoMap, tempoMapOffset):
        pos1 = tempoMapOffset
        pos2 = pos1 + duration
        fduration =  GetTempoMap(tempoMap, pos2)- GetTempoMap(tempoMap, pos1)
        self.EnginePlayBeat (engine, buf, fduration)


    def PlayBeatB(self, engine, buf, duration, tempo):
        fduration=abs(duration*60000)/(tempo*48)
        self.EnginePlayBeat (engine, buf, fduration)

    @staticmethod
    def PlaySilenceA(buf, duration, tempoMap, tempoMapOffset):
        buf.setCursor(GetTempoMap(tempoMap, tempoMapOffset+duration))

    @staticmethod
    def PlayBackspaceA(buf, duration, tempoMap, tempoMapOffset):
        buf.setCursor(GetTempoMap(tempoMap, tempoMapOffset-duration))

    @staticmethod
    def PlaySilenceB(buf, duration, tempo):
        fduration=duration*60000/(tempo*48)
        buf.moveCursor(fduration)

    @staticmethod
    def PlayBackspaceB(buf, duration, tempo):
        fduration=duration*60000/(tempo*48)
        buf.moveCursor(-fduration)


def PlaySequence(perc_list, buf, seq, tempo):
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
        percId = item[0]
        operation = item[1]
        if isNumber(operation):
            duration = operation

            if using_tempo_map:
                if percId >= 0:
                    perc_list[percId].shell.PlayBeatA(perc_list[percId].engine, buf, duration, tempo_map, beatPos)
                elif duration >=0:
                    PercussionShell.PlaySilenceA(buf, duration, tempo_map, beatPos)
                else:
                    PercussionShell.PlayBackspaceA(buf, -duration, tempo_map, beatPos)
            else:
                if percId >= 0:
                    perc_list[percId].shell.PlayBeatB(perc_list[percId].engine, buf, duration, tempo)
                elif duration >=0:
                    PercussionShell.PlaySilenceB(buf, duration, tempo)
                else:
                    PercussionShell.PlayBackspaceB(buf, -duration, tempo)
            beatPos+=duration
        elif type(operation)== str:
            perc_list[percId].tune(operation)

class Percussion:
    def __init__(self):
        self.shell=PercussionShell()
        
    @staticmethod
    def play(percList, buf, seq, tempo=80.0):
        PlaySequence(percList, buf, seq, tempo)

    def tune(self,cmd):
        if not self.shell.tune(cmd):
            self.engine.tune(cmd)
    def setBeatVolume(self,volume):
        self.shell.volume=volume
    def setBeatPan(self,pan):
        self.shell.pan=pan





