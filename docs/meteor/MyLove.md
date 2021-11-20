<script type="text/javascript" src="meteor.js"></script>

<script type="text/javascript">
    window.onload = function() {
        var restiktok = {
            canvasID : "canvastiktok",
            audioID : "audiotiktok",
            dataPath : "MyLove.meteor"
        };
        meteortiktok = new Meteor(restiktok);
    };
</script>

<style type="text/css">
    canvas {
        display: block;
        width: 100%;
        height: width*0.5625;
    }		
</style>

<div>
    <canvas id="canvastiktok" width="800" height="450"></canvas>
</div>
<div>
    <audio id='audiotiktok' controls="controls">
        <source type="audio/mpeg" src="MyLove.mp3"/>
    </audio>
</div>

# My Love

## Info
* Originally sung by: Westlife
* Melody/Lyric by: Jörgen Elofsson, Per Magnusson, David Kreuger and Pelle Nylén
* CZloid voicebank: [http://utau.wiki/utau:czloid](http://utau.wiki/utau:czloid)
* Video published earlier on Bilibili: [https://www.bilibili.com/video/av19520577](https://www.bilibili.com/video/av19520577)
* Teto English Voicebank (CVVC) Voicebank: [http://kasaneteto.jp/en/voicebank.html](http://kasaneteto.jp/en/voicebank.html) 

```python
import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.MeteorDocument()
doc.setTempo(72)

Teto = ScoreDraft.TetoEng_UTAU()
Teto.setLyricConverter(ScoreDraft.TTEnglishConverter)

Guitar=ScoreDraft.SteelGuitar()
Bass=ScoreDraft.Bass()

track=doc.newBuf()
track_g=doc.newBuf()
track_b=doc.newBuf()
track_mix=doc.newBuf()


FreqsC=Freqs[:]
FreqsD=[f*Freqs[2] for f in Freqs]

def doS(octave=5, duration=48):
	return note(octave,Freqs[1],duration)

def faS(octave=5, duration=48):
	return note(octave,Freqs[6],duration)

def soS(octave=5, duration=48):
	return note(octave,Freqs[8],duration)

def Chord3(A,octA,B,octB,C,octC, duration, delay):
	return [A(octA,duration), BK(duration-delay), B(octB,duration-delay), BK(duration-delay*2), C(octC,duration-delay*2)]

def MiSoDo(duration, delay=0):
	return Chord3(mi,4,so,4,do,5,duration,delay)

def DoMiSo(duration, delay=0):
	return Chord3(do,5,mi,5,so,5,duration,delay)

def TiReSo(duration, delay=0):
	return Chord3(ti,4,re,5,so,5,duration,delay)

def TiMiSo(duration, delay=0):
	return Chord3(ti,4,mi,5,so,5,duration,delay)

def LaDoMi(duration, delay=0):
	return Chord3(la,4,do,5,mi,5,duration,delay)

def LaDoFa(duration, delay=0):
	return Chord3(la,4,do,5,fa,5,duration,delay)

def DoReSo(duration, delay=0):
	return Chord3(do,5,re,5,so,5,duration,delay)

def LaReFa(duration, delay=0):
	return Chord3(la,4,re,5,fa,5,duration,delay)

def FaLaDoH(duration, delay=0):
	return Chord3(fa,5,la,5,do,6,duration,delay)

def MiSoDoH(duration, delay=0):
	return Chord3(mi,5,so,5,do,6,duration,delay)

def LaDoFaH(duration, delay=0):
	return Chord3(la,5,do,6,fa,6,duration,delay)

def SoDoMiH(duration, delay=0):
	return Chord3(so,5,do,6,mi,6,duration,delay)

def LaDoMiH(duration, delay=0):
	return Chord3(la,5,do,6,mi,6,duration,delay)

def FaSLaReH(duration, delay=0):
	return Chord3(faS,5,la,5,re,6,duration,delay)

def TiReSoH(duration, delay=0):
	return Chord3(ti,5,re,6,so,6,duration,delay)

def SoTiReH(duration, delay=0):
	return Chord3(so,5,ti,5,re,6,duration,delay)

def SoSTiMiH(duration, delay=0):
	return Chord3(soS,5,ti,5,mi,6,duration,delay)

def MiLaDoH(duration, delay=0):
	return Chord3(mi,5,la,5,do,6,duration,delay)

def ReSoTi(duration, delay=0):
	return Chord3(re,5,so,5,ti,5,duration,delay)

def ReSoDoH(duration, delay=0):
	return Chord3(re,5,so,5,do,6,duration,delay)

def DoFaLa(duration, delay=0):
	return Chord3(do,5,fa,5,la,5,duration,delay)

seq=[BL(192)]
seq_g=[BL(96), fa(5,12), la(4,12), re(5,12), fa(5,60), BK(48), do(5,24), la(4,24)]
seq_b=[BL(96), re(4,48), do(4,48)]

seq+=[BL(192)]
seq_g+=[re(5,48), BK(48), so(4,48), re(5,24), BK(24), so(4,24), do(5,12), ti(4,12), la(4,48), la(4,24), ti(4,24)]
seq_b+=[ti(3,72), la(3,12), so(3,12), fa(3,96), BK(72), do(4,12), fa(4,60)]

seq+=[BL(96), BL(72), BL(12)]
seq_g+=MiSoDo(48)+MiSoDo(36)+MiSoDo(24)+MiSoDo(12)+MiSoDo(24)+MiSoDo(24)+[BL(24)]
seq_b+=[do(3,144), BL(48)]

seq+=[('{n', so(5,12), 'Em', do(6,24),'pti', do(6,12), 'strit', do(6, 36)), BL(12)]
seq+=[('{n',so(5,12), 'Em', re(6,24),'pti', re(6,12), 'haUs',re(6, 36)), BL(12)]
seq_g+=DoMiSo(48, 2)+DoMiSo(48, 2)+TiReSo(48,2)+TiReSo(48,2)
seq_b+=[do(4,72), do(4,24),so(3,72),so(3,24)]

seq+=[('@', so(5,12), 'hoUl', ti(5,24), 'In', ti(5,12), 'saId', ti(5,24), 'maI',do(6,24),'hArt',do(6,48)), BL(48)]
seq_g+=TiMiSo(48,2)+TiMiSo(48,2)+LaDoMi(48,2)+LaDoMi(24)+LaDoMi(12)+LaDoMi(12)
seq_b+=[mi(3,72),mi(3,24), la(3,96)]

seq+=[('aIm', so(5,12), 'Ol', la(5,24), '@l', la(5,12),'oUn', la(5,36)),BL(12)]
seq+=[('D@',la(5,12),'rum',la(5,24),'zAr',la(5,24), 'gEt', la(5,12),'IN', la(5,24), 'smOl', la(5,24), '3', so(5,36))]
seq_g+=LaDoFa(48,2)+LaDoFa(24)+LaDoFa(12)+LaDoFa(12)+LaDoFa(48,2)+LaDoFa(48,2)
seq_b+=[fa(3,96),fa(3,72), fa(3,24)]

seq+=[BL(48),BL(48), BL(36)]
seq_g+=DoReSo(48,2)+DoReSo(24)+DoReSo(12)+DoReSo(12)+TiReSo(48,2)+TiReSo(24)+TiReSo(12)+TiReSo(12)
seq_b+=[so(3,96),so(3,96)]

seq+=[('aI',so(5,12),'w@n',do(6,24),'d3',do(6,12),'haU',do(6,36)), BL(12)]
seq+=[('aI',so(5,12),'w@n',re(6,24),'d3',re(6,12),'waI',re(6,36)), BL(12)]
seq_g+=DoMiSo(48, 2)+DoMiSo(48, 2)+TiReSo(48,2)+TiReSo(48,2)
seq_b+=[do(4,72), do(4,24),so(3,72),so(3,24)]

seq+=[('aI',mi(6,12), 'w@n', so(6,24), 'd3', so(6,12), 'wEr', so(6,12), mi(6,12), 'DeI', mi(6,12), do(6,12), 'Ar', do(6,48)), BL(48)]
seq_g+=TiMiSo(48,2)+TiMiSo(48,2)+LaDoMi(48,2)+LaDoMi(24)+LaDoMi(12)+LaDoMi(12)
seq_b+=[mi(3,72),mi(3,24), la(3,48), so(3,48)]

seq+=[('D@',so(5,12),'deIz',la(5,24), 'wi',la(5,12), 'h{d', la(5,36)), BL(12)]
seq+=[('D@',fa(5,12),'sONz',la(5,24), 'wi',la(5,12), 's{N', do(6,36), 't@g', la(5,12), 'ED', la(5,12), so(5,12), '3', so(5,36)), BL(12)]
seq_g+=LaDoFa(48,2)+LaDoFa(48,2)+LaDoFa(24)+[fa(4,24), la(4,48), BK(48), re(5,48), BK(48), faS(5,48)]
seq_b+=[fa(3,72),fa(3,24), fa(3,48), faS(3,48)]

seq+=[('oU', ti(5,12),la(5,12), 'j{', ti(5,6), la(5,6), so(5,48)), BL(24)]
seq_g+=DoReSo(48,2)+DoReSo(48,2)+TiReSo(48,2)+TiReSo(48,2)
seq_b+=[so(3,72),so(3,24),so(3,96)]

seq+=[('@n',so(5,24),'dol',fa(6,48),'maI',la(5,48),'l@v',ti(5,48)), BL(36)]
seq_g+=[fa(5,12),la(4,12),re(5,12),fa(5,60),BK(48),do(5,24),la(4,24)]+Chord3(so,4,re,5,so,5,96,4)
seq_b+=[re(4,48),do(4,48),ti(3,48),ti(3,48)]

seq+=[('aIm',so(5,12),'hoU',do(6,24),'ldIN',do(6,12),'An',do(6,36),'f3',mi(6,12),'Ev',re(6,12),do(6,12),'3',do(6,36)),BL(48)]
seq_g+=[so(4,48),BK(46),mi(5,46),do(5,48),BK(46),so(5,46)]+LaDoFa(48,2)+LaDoFa(48,2)
seq_b+=[do(4,48),mi(3,48),fa(3,48),fa(3,24),mi(3,24)]

seq+=[('ri',fa(6,24),'tSIN',fa(6,12),'fOr',fa(6,36),'D@',fa(6,12),'l@v',fa(6,36),'D{t',mi(6,12),'simz',do(6,24),'soU',re(6,36),'fAr',re(6,96)), BL(48)]
seq_g+=LaReFa(48,2)+LaReFa(48,2)+LaDoFa(48,2)+LaReFa(48,2)
seq_b+=[re(3,72),mi(3,24),fa(3,96)]
seq_g+=DoReSo(48,2)+DoReSo(48,2)+TiReSo(96,4)
seq_b+=[so(3,72),so(3,24),so(3,96)]

seq+=[('soU',do(6,24),'aI',ti(5,24),'seI',la(5,24),'@',so(5,12),'lIt',so(5,24),'@l',so(5,24),'prEr',so(5,36)), BL(12)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('@nd',so(5,12),'hoUp',do(6,24),'maI',ti(5,24),'drimz',la(5,24),'wIl',so(5,12),'teIk',so(5,24),'mi',fa(5,24),'DEr',mi(5,36)), BL(24)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,24),'skaIz',fa(6,24),'Ar',mi(6,12),'blu',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,60),'maI',la(5,24),'l@v',ti(5,48)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(48,2)+FaSLaReH(48,2)+TiReSoH(48,2)+TiReSoH(48,2)
seq_b+=[faS(3,72),faS(3,24),so(3,72),so(3,24)]

seq+=[('oU',do(6,24),'v3',ti(5,24),'siz',la(5,24),'fr@m',so(5,12),'koUst',so(5,36),'u',fa(5,12),'koUst',so(5,36)),BL(12)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('tu',so(5,12),'faInd',do(6,24),'D@',ti(5,12),'pleIs',la(5,36),'aI',so(5,12),'l@v',so(6,24),fa(6,12),'D@',fa(6,12),'moUst',mi(6,36)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,12),'fild',fa(6,36),'zAr',mi(6,12),'grin',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,108)),BL(48)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(48,2)+FaSLaReH(24)+SoTiReH(120,4)
seq_b+=[re(4,72),so(3,72),so(3,48)]

seq+=[('maI',ti(5,24),'l@v',do(6,168)), BL(36)]
seq_g+=MiSoDoH(192,4)
seq_b+=[do(3,192)]

seq+=[('aI', so(5,12), 'traI', do(6,24),'tu', do(6,12), 'rEd', do(6, 36)), BL(12)]
seq+=[('aI',so(5,12), 'goU', re(6,24),'tu', re(6,12), 'w3k',re(6, 36)), BL(12)]
seq_g+=DoMiSo(48, 2)+DoMiSo(48, 2)+TiReSo(48,2)+TiReSo(48,2)
seq_b+=[do(4,72), do(4,24),so(3,72),so(3,24)]

seq+=[('aIm', so(5,12), 'l{f', ti(5,24), 'IN', ti(5,12), 'wID', ti(5,24), 'maI',do(6,24),'frEndz',do(6,48)), BL(48)]
seq_g+=TiMiSo(48,2)+TiMiSo(48,2)+LaDoMi(48,2)+LaDoMi(24)+LaDoMi(12)+LaDoMi(12)
seq_b+=[mi(3,72),mi(3,24), la(3,96)]

seq+=[('b@t', so(5,12), 'aI', la(5,24), 'k{nt', la(5,12),'stAp', la(5,36)),BL(12)]
seq+=[('tu',la(5,12),'kip',la(5,24),'maI',la(5,24), 'sElf', la(5,12),'r@m', la(5,24), 'TIN', la(5,24), 'kIN', so(5,36))]
seq_g+=LaDoFa(48,2)+LaDoFa(24)+LaDoFa(12)+LaDoFa(12)+LaDoFa(48,2)+LaDoFa(48,2)
seq_b+=[fa(3,96),fa(3,72), fa(3,24)]

seq+=[BL(12),('oU', ti(5,12),la(5,12), 'noU', ti(5,6), la(5,6), so(5,48)), BL(36)]
seq_g+=DoReSo(48,2)+DoReSo(24)+DoReSo(12)+DoReSo(12)+TiReSo(48,2)+TiReSo(24)+TiReSo(12)+TiReSo(12)
seq_b+=[so(3,96),so(3,96)]

seq+=[('aI',so(5,12),'w@n',do(6,24),'d3',do(6,12),'haU',do(6,36)), BL(12)]
seq+=[('aI',so(5,12),'w@n',re(6,24),'d3',re(6,12),'waI',re(6,36)), BL(12)]
seq_g+=DoMiSo(48, 2)+DoMiSo(48, 2)+TiReSo(48,2)+TiReSo(48,2)
seq_b+=[do(4,72), do(4,24),so(3,72),so(3,24)]

seq+=[('aI',mi(6,12), 'w@n', so(6,24), 'd3', so(6,12), 'wEr', so(6,12), mi(6,12), 'DeI', mi(6,12), do(6,12), 'Ar', do(6,48)), BL(48)]
seq_g+=TiMiSo(48,2)+TiMiSo(48,2)+LaDoMi(48,2)+LaDoMi(24)+LaDoMi(12)+LaDoMi(12)
seq_b+=[mi(3,72),mi(3,24), la(3,48), so(3,48)]

seq+=[('D@',so(5,12),'deIz',la(5,24), 'wi',la(5,12), 'h{d', la(5,36)), BL(12)]
seq+=[('D@',fa(5,12),'sONz',la(5,24), 'wi',la(5,12), 's{N', do(6,36), 't@g', la(5,12), 'ED', la(5,12), so(5,12), '3', so(5,36)), BL(12)]
seq_g+=LaDoFa(48,2)+LaDoFa(48,2)+LaDoFa(24)+[fa(4,24), la(4,48), BK(48), re(5,48), BK(48), faS(5,48)]
seq_b+=[fa(3,72),fa(3,24), fa(3,48), faS(3,48)]

seq+=[('oU', ti(5,12),la(5,12), 'j{', ti(5,6), la(5,6), so(5,48)), BL(24)]
seq_g+=DoReSo(48,2)+DoReSo(48,2)+TiReSo(48,2)+TiReSo(48,2)
seq_b+=[so(3,72),so(3,24),so(3,96)]

seq+=[('@n',so(5,24),'dol',fa(6,48),'maI',la(5,48),'l@v',ti(5,48)), BL(36)]
seq_g+=[fa(5,12),la(4,12),re(5,12),fa(5,60),BK(48),do(5,24),la(4,24)]+Chord3(so,4,re,5,so,5,96,4)
seq_b+=[re(4,48),do(4,48),ti(3,48),ti(3,48)]

seq+=[('aIm',so(5,12),'hoU',do(6,24),'ldIN',do(6,12),'An',do(6,36),'f3',mi(6,12),'Ev',re(6,12),do(6,12),'3',do(6,36)),BL(48)]
seq_g+=[so(4,48),BK(46),mi(5,46),do(5,48),BK(46),so(5,46)]+LaDoFa(48,2)+LaDoFa(48,2)
seq_b+=[do(4,48),mi(3,48),fa(3,48),fa(3,24),mi(3,24)]

seq+=[('ri',fa(6,24),'tSIN',fa(6,12),'fOr',fa(6,36),'D@',fa(6,12),'l@v',fa(6,36),'D{t',mi(6,12),'simz',do(6,24),'soU',re(6,36),'fAr',re(6,96)), BL(48)]
seq_g+=LaReFa(48,2)+LaReFa(48,2)+LaDoFa(48,2)+LaReFa(48,2)
seq_b+=[re(3,72),mi(3,24),fa(3,96)]
seq_g+=DoReSo(48,2)+DoReSo(48,2)+TiReSo(96,4)
seq_b+=[so(3,72),so(3,24),so(3,96)]

seq+=[('soU',do(6,24),'aI',ti(5,24),'seI',la(5,24),'@',so(5,12),'lIt',so(5,24),'@l',so(5,24),'prEr',so(5,36)), BL(12)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('@nd',so(5,12),'hoUp',do(6,24),'maI',ti(5,24),'drimz',la(5,24),'wIl',so(5,12),'teIk',so(5,24),'mi',fa(5,24),'DEr',mi(5,36)), BL(24)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,24),'skaIz',fa(6,24),'Ar',mi(6,12),'blu',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,60),'maI',la(5,24),'l@v',ti(5,48)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(48,2)+FaSLaReH(48,2)+TiReSoH(48,2)+TiReSoH(48,2)
seq_b+=[faS(3,72),faS(3,24),so(3,72),so(3,24)]

seq+=[('oU',do(6,24),'v3',ti(5,24),'siz',la(5,24),'fr@m',so(5,12),'koUst',so(5,36),'u',fa(5,12),'koUst',so(5,36)),BL(12)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('tu',so(5,12),'faInd',do(6,24),'D@',ti(5,12),'pleIs',la(5,36),'aI',so(5,12),'l@v',so(6,24),fa(6,12),'D@',fa(6,12),'moUst',mi(6,36)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,12),'fild',fa(6,36),'zAr',mi(6,12),'grin',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,72)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]

seq+=[('tu',la(5,12),'hoUl',mi(6,24),'dju',re(6,12),'In',do(6,36),'maI',ti(5,12),'Armz',do(6,60)), BL(36)]
seq_g+=FaSLaReH(96,4)+SoSTiMiH(48,2)+[ti(5,48), BK(46), mi(6,46)]
seq_b+=[re(4,72),re(4,24),mi(4,48), soS(3,48)]

seq+=[('tu',la(5,12),'prAm',re(6,24),'@s',do(6,12),'ju',ti(5,24),'maI',la(5,24),'l@v',ti(5,24),la(5,12),so(5,24)), BL(36)]
seq_g+=MiLaDoH(48,2)+MiLaDoH(48,2)+[re(5,48),BK(46),la(5,46),re(5,48),BK(46),la(5,46)]
seq_b+=[la(3,72),la(3,24),faS(3,72),faS(3,24)]

seq+=[('tu',la(5,12),'tEl',mi(6,24),'ju',re(6,12),'fr@m',do(6,36),'D@',ti(5,12),'hArt',do(6,60)), BL(36)]
seq_g+=ReSoTi(48,2)+ReSoTi(48,2)+[soS(5,48),mi(5,48),BK(96),mi(6,24),re(6,12),do(6,36),ti(5,24)]
seq_b+=[so(3,72),so(3,24),mi(3,48),soS(3,48)]

seq+=[('jUr',mi(5,12),'Ol',la(5,24),'aIm',la(5,12),'TIN',la(5,36),'kIN',so(5,12),'Ov',so(5,156)),BL(48)]
seq_g+=MiLaDoH(48,2)+MiLaDoH(48,2)+[re(5,48),BK(46),la(5,46),re(5,48),BK(46),la(5,46)]
seq_b+=[la(3,72), la(3,24),faS(3,72),faS(3,24)]
seq_g+=ReSoDoH(48,2)+ReSoDoH(48,2)
seq_g+=[re(5,96), BK(92), so(5,92), BK(90), ti(5,42), la(5,24), ti(5,24)]
seq_b+=[so(3,72), re(4,24), so(3,96)]

seq+=[BL(192)]
seq_g+=[mi(5,96), BK(92), so(5,92), BK(92), do(6,20), do(6,12), do(6,48), do(6,12), so(5,36), ti(5,60), BK(96), do(6,12), re(6,12), re(6,12), re(6,60)]
seq_b+=[do(4,96), so(3,96)]

seq+=[BL(192-12)]
seq_g+=[so(5,84), mi(5,108), BK(192), ti(5,24), ti(5,12), ti(5,24), do(6,24), do(6,60), ti(5,48)]
seq_b+=[mi(3,96), la(3,48), so(3,48)]

seq+=[('aIm',fa(5,12), 'ri',fa(6,24),'tSIN',fa(6,12),'fOr',fa(6,36),'D@',fa(6,12), 'l@v',fa(6,24),mi(6,12),'D{t',mi(6,12), 'simz',do(6,24),'soU',re(6,36), 'fAr', re(6,72)), BL(24)]
seq_g+=DoFaLa(48,2)+DoFaLa(24)+DoFaLa(24)+DoFaLa(48,2)+DoFaLa(24)+DoFaLa(24)
seq_b+=[fa(3,96),fa(3,96)]

seq+=[('soU',re(6,48),'aI',re(6,24),doS(6,24))]
seq_g+=ReSoDoH(48,2)+ReSoDoH(48,2)+ReSoTi(24)+[re(5,12),mi(5,12),faS(5,10),so(5,10),la(5,10),ti(5,9),doS(6,9)]
seq_b+=[so(3,96),so(3,24), BL(24), BL(48)]

# To D
Freqs[:]=FreqsD

seq+=[('seI',la(5,24),'@',so(5,12),'lIt',so(5,24),'@l',so(5,24),'prEr',so(5,36)), BL(12)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('@nd',so(5,12),'hoUp',do(6,24),'maI',ti(5,24),'drimz',la(5,24),'wIl',so(5,12),'teIk',so(5,24),'mi',fa(5,24),'DEr',mi(5,36)), BL(24)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,24),'skaIz',fa(6,24),'Ar',mi(6,12),'blu',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,60),'maI',la(5,24),'l@v',ti(5,48)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(48,2)+FaSLaReH(48,2)+TiReSoH(48,2)+TiReSoH(48,2)
seq_b+=[faS(3,72),faS(3,24),so(3,72),so(3,24)]

seq+=[('oU',do(6,24),'v3',ti(5,24),'siz',la(5,24),'fr@m',so(5,12),'koUst',so(5,36),'u',fa(5,12),'koUst',so(5,36)),BL(12)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('tu',so(5,12),'faInd',do(6,24),'D@',ti(5,12),'pleIs',la(5,36),'aI',so(5,12),'l@v',so(6,24),fa(6,12),'D@',fa(6,12),'moUst',mi(6,36)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,12),'fild',fa(6,36),'zAr',mi(6,12),'grin',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,108)),BL(48)]
seq+=[('soU',do(6,24),'aI',ti(5,24))]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(48,2)+FaSLaReH(24)+[so(5,120), BK(116),ti(5,68),BK(64),re(6,64), do(6,24), ti(5,24)]
seq_b+=[re(4,72),so(3,72),so(3,48)]

seq+=[('seI',la(5,24),'@',so(5,12),'lIt',so(5,24),'@l',so(5,24),'prEr',so(5,36)), BL(12)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('@nd',so(5,12),'hoUp',do(6,24),'maI',ti(5,24),'drimz',la(5,24),'wIl',so(5,12),'teIk',so(5,24),'mi',fa(5,24),'DEr',mi(5,36)), BL(24)]
seq_g+=FaLaDoH(48,2)+FaLaDoH(48,2)+MiSoDoH(48,2)+MiSoDoH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,24),'skaIz',fa(6,24),'Ar',mi(6,12),'blu',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,60),'maI',la(5,24),'l@v',ti(5,48)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(48,2)+FaSLaReH(48,2)+TiReSoH(48,2)+TiReSoH(48,2)
seq_b+=[faS(3,72),faS(3,24),so(3,72),so(3,24)]

seq+=[('oU',do(6,24),'v3',ti(5,24),'siz',la(5,24),'fr@m',so(5,12),'koUst',so(5,36),'u',fa(5,12),'koUst',so(5,36)),BL(12)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,96)]

seq+=[('tu',so(5,12),'faInd',do(6,24),'D@',ti(5,12),'pleIs',la(5,36),'aI',so(5,12),'l@v',so(6,24),fa(6,12),'D@',fa(6,12),'moUst',mi(6,36)),BL(24)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),do(4,24)]

seq+=[('wEr',mi(6,24),'D@',so(6,12),'fild',fa(6,36),'zAr',mi(6,12),'grin',re(6,36)),BL(12)]
seq+=[('tu',do(6,12),'si',mi(6,24),'ju',re(6,12),'w@n',do(6,36),'s@g',mi(6,12),'En',re(6,108)),BL(48)]
seq_g+=LaDoFaH(48,2)+LaDoFaH(48,2)+SoDoMiH(48,2)+LaDoMiH(48,2)
seq_b+=[fa(3,72),fa(3,24),do(4,72),la(3,24)]
seq_g+=FaSLaReH(72,2)+SoTiReH(120,4)
seq_b+=[faS(3,48),faS(3,24),so(3,120)]

seq+=[('maI',ti(5,24),'l@v',do(6,168)), BL(48)]
seq_g+=[so(4,192), BK(188), do(5,188), BK(184), mi(5,184)]
seq_b+=[do(3,192)]

doc.sing(seq, Teto, track)
doc.playNoteSeq(seq_g, Guitar, track_g)
doc.playNoteSeq(seq_b, Bass, track_b)

doc.setTrackVolume(track_g, 0.5)
doc.setTrackVolume(track_b, 0.5)

doc.meteor()
doc.mixDown('MyLove.wav')
```





