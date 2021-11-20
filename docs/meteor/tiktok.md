<script type="text/javascript" src="meteor.js"></script>

<script type="text/javascript">
    window.onload = function() {
        var restiktok = {
            canvasID : "canvastiktok",
            audioID : "audiotiktok",
            dataPath : "tiktok.meteor"
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
        <source type="audio/mpeg" src="tiktok.mp3"/>
    </audio>
</div>

# Tik Tok

## Info
* Originally sung by: Ke$ha
* Melody/Lyric by: Ke$ha, Lukasz Gottwald, Benny Blanco
* CZloid voicebank: [http://utau.wiki/utau:czloid](http://utau.wiki/utau:czloid)

```python
import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.MeteorDocument()
doc.setTempo(110)

singer = ScoreDraft.CZloid_UTAU()
singer.setLyricConverter(ScoreDraft.VCCVEnglishConverter)
singer.setCZMode()

Guitar=ScoreDraft.CleanGuitar()

BassDrum=ScoreDraft.BassDrum()
Snare=ScoreDraft.Snare()
Clap=ScoreDraft.Clap()

perc_list=[BassDrum,Snare, Clap]

def dong(duration=48):
	return (0,duration)

def cha(duration=48):
	return (1,duration)

def pia(duration=48):
	return (2,duration)

track=doc.newBuf()
track_g=doc.newBuf()
track_p=doc.newBuf()

def Chord(elems, duration, delay=0):
	ret=[]
	for elem in elems:
		ret+=[elem[0](elem[1], duration)]
		duration-=delay
		ret+=[BK(duration)]
	ret+=[BL(duration)]
	return ret

def FaLaDo(duration):
	return Chord([(fa,4), (la,4), (do,5), (fa,5) ], duration)

def SoTiRe(duration):
	return Chord([(so,4), (ti,4), (re,5), (so,5) ], duration)

def LaDoMi(duration):
	return Chord([(la,4), (do,5), (mi,5), (la,5) ], duration)

def ReFaLaDo(duration):
	return Chord([(re,4), (fa,4), (la,4), (do,5)], duration)

FreqsC=Freqs[:]
FreqsF=[f*Freqs[5] for f in Freqs]
Freqs[:]=FreqsF

seq_g = FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq =[BL(192)]

seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq +=[BL(192)]

seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq +=[BL(192)]

seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq +=[BL(192)]

line = ('wA', la(4,24)+do(5,0),'kup', do(5,24), 'in', la(4,24), 'dhx', la(4,24))
line +=('m0r', mi(4,24)+fa(4,0), 'n1ng', fa(4,24)+la(4,0), 'fE', fa(4,24)+mi(4,0), 'l1ng', mi(4,24)+re(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)

line +=('lIk', fa(4,36)+so(4,0), 'pE', so(4,36), 'did', la(4,36)+so(4,0))
seq += [ line, BL(4) ]
line =('p6t', so(4,16)+mi(4,0), 'mI', mi(4,16),re(4,0), 'gla', so(4,24), 'ses', mi(4,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)

line +=('an', fa(4,24)+do(5,0), 'Im', la(4,24)+mi(4,0), '8', fa(4,24)+mi(4,0), 'dhx', fa(4,24)+mi(4,0), 'd0r', fa(4,24)+mi(4,0))
line +=('Im', fa(4,24)+do(5,0), 'g0r', la(4,24)+fa(4,0), 'na', mi(4,24)+do(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)

line +=('hit', ti(4,36)+so(4,0), 'dhi', fa(4,36)+re(4,0), 'si', so(4,24), 'ti', mi(4,24)+do(4,0))
seq += [line]
line = ('bE',la(4,24)+do(5,0),'f0r',do(5,24), 'I', do(4,24)+mi(4,0))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)

line += ('lEv', fa(4,36)+do(5,0))
seq += [line, BL(12)]
line = ('brush', la(4,24)+do(5,0), 'mI', do(5,24), 'tEth', la(4,36)+fa(4,0))
seq += [line, BL(12)]
line = ('with', fa(4,24)+mi(4,0), 'x', mi(4,24),do(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)

line += ('ba', re(4,24)+so(4,0), 't9l', so(4,24), 'xv', so(4,24), 'j@k',la(4,36)+mi(4,0))
seq += [line, BL(12)]
line = ('k0rz', so(4,24)+mi(4,0), 'wen', mi(4,24), 'I', mi(4,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)

line += ('lEv', fa(4,36)+do(5,0))
seq += [line, BL(12)]
line = ('f0r', la(4,24), 'thx', la(4,24), 'nIt', la(4,24)+do(4,0), 'I', do(4,24)+la(4,0), 'ent', la(4,36)+do(4,0))
seq += [line, BL(12)]
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)

line = ('ku', la(4,24)+do(4,0), 'm1ng', mi(4,48)+do(4,0), 'b@k', la(4,36)+do(4,0))
seq_g+=[BL(192)]
seq += [line, BL(12)]
line = ('Im', do(4,24)+la(4,0), 't9l', mi(4,24)+do(4,0), 'k1ng', mi(4,24)+do(4,0))

p_skip=ScoreDraft.TellDuration(seq_g)
perc_loop=[dong(48), pia(24), dong(48), BL(24), pia(48)]

line +=('pe',fa(4,12), 'di', fa(4,12), 'ky3r', fa(4,24)+do(5,0))
line +=('an',la(4,24)+fa(4,0), '8r', fa(4,24)+do(4,0), 'tOs', fa(4,48)+do(4,0), 'tOs', fa(4,36)+do(4,0))
seq += [line,BL(12)]
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p =[BL(p_skip)]+perc_loop

line =('trI',fa(4,12), 'y1ng', fa(4,12), 'an', fa(4,24)+do(5,0))
line +=('9l',la(4,24)+fa(4,0), '8r', fa(4,24)+do(4,0), 'klOth', fa(4,48)+do(4,0), 'klOth', fa(4,36)+do(4,0))
seq += [line]
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line =('bQs',fa(4,12), 'blO', fa(4,24), '1ng', fa(4,24)+do(5,0))
line +=('up',la(4,24)+fa(4,0), '8r', fa(4,24)+do(4,0), 'f9ns', fa(4,48)+do(4,0), 'f9ns', fa(4,48)+do(4,0))
seq += [line, BL(120)]
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p +=perc_loop

line = 	('drap',la(4,24)+do(4,0), 'ta',la(4,24)+do(4,0), 'p1ng', mi(4,24)+do(4,0))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line += ('plA',fa(4,12), 'y1ng', fa(4,12), '8r', fa(4,24)+do(5,0))
line += ('fA', la(4,12)+so(4,0), 'vx', so(4,12)+fa(4,0), 'rit', fa(4,24)+do(4,0), 'sE', fa(4,48)+do(4,0), 'dEs', fa(4,36)+do(4,0))
seq += [line, BL(12)]
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p +=perc_loop

line =('p6',fa(4,12), 'l1ng', fa(4,12), 'up', fa(4,24)+do(5,0))
line +=('t6',la(4,24)+fa(4,0), 'dhx', fa(4,24)+do(4,0), 'par', fa(4,48)+do(4,0), 'tis', fa(4,36)+do(4,0))
seq += [line, BL(12)]
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line =('trIng',fa(4,12),'t6', fa(4,12), 'get', fa(4,24)+do(5,0))
line +=('li',la(4,12)+so(4,0), 't9l', so(4,12)+fa(4,0), 'bit', fa(4,24)+do(4,0), 'tip', fa(4,48)+do(4,0))
seq += [line, BL(48)]
seq_g += ReFaLaDo(36)+ReFaLaDo(36)+ReFaLaDo(24)+[BL(72)]+ReFaLaDo(24)
seq_p +=perc_loop

seq += [('sE', fa(4,48)+do(4,0)), BL(96)]
line = ('dant', do(5,24), re(5,24))
seq_g +=[BL(48)]+ SoTiRe(48)+ SoTiRe(48)+ SoTiRe(48)
seq_p +=[BL(144), cha(24), cha(24)]

perc_loop=[dong(),cha(), dong(), cha()]

line += ('stap', do(5,48), 'mAk', do(5,24), 'it', do(5,24), 'pap', do(5,36))
seq += [line, BL(12)]
line = ('dE', do(5,24), 'jA', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('blO', do(5,12), re(5,12), 'mI', do(5,24), 'spE', do(5,24), 'k3s', do(5,24), 'up', do(5,36))
seq += [line, BL(12)]
line = ('t6', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('nIt', do(5,48), 'Im', do(5,24), 'a', do(5,24), 'fIt', do(5,36))
seq += [line, BL(12)]
line = ('til', do(5,24), 'wE', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('sE', do(5,12), re(5,12), 'dhx', do(5,24), 'sun', ti(4,24), la(4,24), 'lIt', la(4,36))
seq += [line, BL(12)]
line = ('tik', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('tak', do(5,48), 'an', do(5,24), 'dhx', do(5,24), 'klak', do(5,36))
seq += [line, BL(12)]
line = ('bu', do(5,24), 'dhx', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line += ('par', do(5,24), 'ti', do(5,24), 'dant', do(5,48), 'stap', do(5,24), la(4,24), 'nO', la(4,48))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,48))
seq += [line, BL(48)]
seq_g += ReFaLaDo(36)+ReFaLaDo(36)+ReFaLaDo(24)+[BL(72)]+ReFaLaDo(24)
seq_p +=perc_loop

line = ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,36))
seq += [line, BL(12)]
line = ('dant', do(5,24), re(5,24))
seq_g +=LaDoMi(36)+LaDoMi(36)+SoTiRe(48)+SoTiRe(36)+SoTiRe(36)
seq_p +=perc_loop

line += ('stap', do(5,48), 'mAk', do(5,24), 'it', do(5,24), 'pap', do(5,36))
seq += [line, BL(12)]
line = ('dE', do(5,24), 'jA', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('blO', do(5,12), re(5,12), 'mI', do(5,24), 'spE', do(5,24), 'k3s', do(5,24), 'up', do(5,36))
seq += [line, BL(12)]
line = ('t6', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('nIt', do(5,48), 'Im', do(5,24), 'a', do(5,24), 'fIt', do(5,36))
seq += [line, BL(12)]
line = ('til', do(5,24), 'wE', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('sE', do(5,12), re(5,12), 'dhx', do(5,24), 'sun', ti(4,24), la(4,24), 'lIt', la(4,36))
seq += [line, BL(12)]
line = ('tik', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('tak', do(5,48), 'an', do(5,24), 'dhx', do(5,24), 'klak', do(5,36))
seq += [line, BL(12)]
line = ('bu', do(5,24), 'dhx', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line += ('par', do(5,24), 'ti', do(5,24), 'dant', do(5,48), 'stap', do(5,24), la(4,24), 'nO', la(4,48))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,48))
seq += [line, BL(48)]
seq_g += ReFaLaDo(36)+ReFaLaDo(36)+ReFaLaDo(24)+[BL(72)]+ReFaLaDo(24)
seq_p +=perc_loop

line = ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,24))
line += ('ent', so(4,24)+ti(4,0), 'gat', re(4,24)+so(4,0), 'a', so(4,24)+re(4,0))
seq_g +=LaDoMi(36)+LaDoMi(36)+SoTiRe(48)+SoTiRe(36)+SoTiRe(36)
seq_p +=perc_loop

perc_loop=[dong(), dong(),dong(), dong()]

line +=('ke3r',fa(4,36)+la(4,0)) 
seq += [line, BL(12)]
line = ('in', la(4,24), 'dhx', la(4,24), 'w0rld',la(4,36)+fa(4,0))
seq += [line, BL(12)]
line =('but', fa(4,24)+do(4,0), 'gat', fa(4,24)+do(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p +=perc_loop

line +=('plen',re(4,24)+so(4,0), 'ti', so(4,24), 'xv', so(4,24), 'bEr', ti(4,36)+re(4,0))
seq += [line,BL(12)]
line = ('ent', la(4,24)+do(5,0), 'gat', mi(4,24)+la(4,0), 'nO', mi(4,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line += ('ma', la(4,24), 'ni', la(4,24), 'in', fa(4,24)+do(4,0), 'mI', fa(4,24)+do(4,0), 'pa', la(4,24), 'ket', la(4,24), 'but', la(4,24)+fa(4,0), 'Im', fa(4,24)+do(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p +=perc_loop

line +=('9l',ti(3,24)+re(4,0), 're', re(4,24)+so(4,0), 'di', so(4,24)+re(4,0), 'hEr', ti(4,36)+re(4,0))
seq += [line,BL(12)]
line = ('@nd', la(4,24)+do(5,0), 'n8', mi(4,24)+la(4,0), 'dhx', do(4,24)+mi(4,0))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line +=('dyO',la(4,24)+do(5,0), 'sa', do(5,24), 'lIn',fa(4,24)+la(4,0), '1ng', la(4,24), 'up', fa(4,36)+la(4,0))
seq += [line,BL(12)]
line =('k0rz',la(4,24)+fa(4,0), 'dhA', fa(4,24)+mi(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p +=perc_loop

line+=('hEr', so(4,24)+ti(4,0), 'wE', re(4,24)+so(4,0), 'gats', so(4,48)+ti(4,0), 'w@', la(4,24), 'g3r', la(4,24)+mi(4,0))
line+=('but',la(4,24)+fa(4,0), 'wE', fa(4,24)+mi(4,0))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line+=('ki',fa(4,24)+re(4,0), 'kAm', fa(4,24)+la(4,0), 't6', la(4,24), 'dhx', la(4,24), 'k3rb', la(4,24)+fa(4,0))
line+=('un',fa(4,24)+re(4,0), 'les', la(4,24)+fa(4,0), 'dhA', fa(4,24)+re(4,0))
seq_g += ReFaLaDo(36)+ReFaLaDo(36)+ReFaLaDo(24)+[BL(72)]+ReFaLaDo(24)
seq_p +=perc_loop

line+=('l6k', fa(4,24)+la(4,0), 'lIk', fa(4,24)+la(4,0), 'mIk', fa(4,24)+la(4,0), 'j@', la(4,24)+fa(4,0), 'g3r', fa(4,24)+re(4,0))
seq += [line]
line=('Im', do(4,24)+mi(4,0), 't9l', la(4,12), 'k1ng', la(4,12), 'b8t', mi(4,24)+la(4,0))
seq_g += [BL(192)]
seq_p += [BL(192)]

perc_loop=[dong(48), pia(24), dong(48), BL(24), pia(48)]

line +=('ev',fa(4,12), 'ri', fa(4,12), 'ba', fa(4,12)+la(4,0), 'di', la(4,12)+do(5,0))
line +=('ge',la(4,24)+fa(4,0), 't1ng', fa(4,24)+do(4,0), 'krunk', fa(4,48)+do(4,0), 'krunk', fa(4,36)+do(4,0))
seq += [line,BL(12)]
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p += perc_loop

line =('bQs',fa(4,24), 'trI', fa(4,12)+do(5,0), 't6', do(5,12))
line +=('tuch',la(4,24)+fa(4,0), 'mI', fa(4,24)+do(4,0), 'junk', fa(4,48)+do(4,0), 'junk', fa(4,24)+do(4,0))
seq += [line]
line =('ga', fa(4,12)+la(4,0), 'na', la(4,12)+do(4,0))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line +=('sm@k',la(4,12), 'him', la(4,12)+fa(4,0), 'if', fa(4,12), 'hi', la(4,12)+fa(4,0), 'ge', fa(4,12)+la(4,0), 't1ng', la(4,12) )
line +=('to', la(4,24)+fa(4,0), 'drunk', fa(4,48)+do(4,0), 'drunk', fa(4,48)+do(4,0))
seq += [line, BL(120)]
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p +=perc_loop

line = 	('n8',la(4,24)+do(4,0), 'n8',la(4,24)+do(4,0), 'wE', la(4,24)+do(4,0))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line +=('gO',fa(4,12), 'in', fa(4,12), 'til', fa(4,12)+la(4,0), 'dhA', la(4,12)+do(5,0))
line +=('kik',la(4,24)+fa(4,0), 'us', fa(4,24)+do(4,0), '8t', fa(4,48)+do(4,0), '8t', fa(4,24)+do(4,0))
seq += [line]
line =('0r', fa(4,12)+la(4,0), 'dhx', la(4,12)+do(4,0))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(24)
seq_p += perc_loop

line +=('px',fa(4,24), 'lEs', fa(4,24)+do(5,0))
line +=('shut',la(4,24)+fa(4,0), 'us', fa(4,24)+do(4,0), 'd8n', fa(4,48)+do(4,0), 'd8n', fa(4,36)+do(4,0))
seq += [line, BL(12)]
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(36)+LaDoMi(36)
seq_p +=perc_loop

line =('px',fa(4,24), 'lEs', fa(4,24)+do(5,0))
line +=('shut',la(4,24)+fa(4,0), 'us', fa(4,24)+do(4,0), 'd8n', fa(4,48)+do(4,0), 'd8n', fa(4,36)+do(4,0))
seq += [line, BL(12)]
seq_g += [BL(192)]
seq_p +=[dong(12), dong(12),dong(24), BL(96), pia(48)]

line =('px',fa(4,24), 'px', fa(4,24))
line +=('shut',la(4,24)+fa(4,0), 'us', fa(4,24)+do(4,0))
seq += [line, BL(48)]
line =('dant', do(5,24), re(5,24))
seq_g += [BL(192)]
seq_p +=[dong(12), dong(12),dong(24), BL(96), pia(48)]

perc_loop=[dong(),cha(), dong(), cha()]

line += ('stap', do(5,48), 'mAk', do(5,24), 'it', do(5,24), 'pap', do(5,36))
seq += [line, BL(12)]
line = ('dE', do(5,24), 'jA', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('blO', do(5,12), re(5,12), 'mI', do(5,24), 'spE', do(5,24), 'k3s', do(5,24), 'up', do(5,36))
seq += [line, BL(12)]
line = ('t6', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('nIt', do(5,48), 'Im', do(5,24), 'a', do(5,24), 'fIt', do(5,36))
seq += [line, BL(12)]
line = ('til', do(5,24), 'wE', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('sE', do(5,12), re(5,12), 'dhx', do(5,24), 'sun', ti(4,24), la(4,24), 'lIt', la(4,36))
seq += [line, BL(12)]
line = ('tik', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('tak', do(5,48), 'an', do(5,24), 'dhx', do(5,24), 'klak', do(5,36))
seq += [line, BL(12)]
line = ('bu', do(5,24), 'dhx', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line += ('par', do(5,24), 'ti', do(5,24), 'dant', do(5,48), 'stap', do(5,24), la(4,24), 'nO', la(4,48))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,48))
seq += [line, BL(48)]
seq_g += ReFaLaDo(36)+ReFaLaDo(36)+ReFaLaDo(24)+[BL(72)]+ReFaLaDo(24)
seq_p +=perc_loop

line = ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,36))
seq += [line, BL(12)]
line = ('dant', do(5,24), re(5,24))
seq_g +=LaDoMi(36)+LaDoMi(36)+SoTiRe(48)+SoTiRe(36)+SoTiRe(36)
seq_p +=perc_loop

line += ('stap', do(5,48), 'mAk', do(5,24), 'it', do(5,24), 'pap', do(5,36))
seq += [line, BL(12)]
line = ('dE', do(5,24), 'jA', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('blO', do(5,12), re(5,12), 'mI', do(5,24), 'spE', do(5,24), 'k3s', do(5,24), 'up', do(5,36))
seq += [line, BL(12)]
line = ('t6', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('nIt', do(5,48), 'Im', do(5,24), 'a', do(5,24), 'fIt', do(5,36))
seq += [line, BL(12)]
line = ('til', do(5,24), 'wE', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line +=('sE', do(5,12), re(5,12), 'dhx', do(5,24), 'sun', ti(4,24), la(4,24), 'lIt', la(4,36))
seq += [line, BL(12)]
line = ('tik', do(5,24), re(5,24))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('tak', do(5,48), 'an', do(5,24), 'dhx', do(5,24), 'klak', do(5,36))
seq += [line, BL(12)]
line = ('bu', do(5,24), 'dhx', do(5,24))
seq_g += FaLaDo(36)+FaLaDo(36)+FaLaDo(24)+[BL(72)]+FaLaDo(12)+FaLaDo(12)
seq_p +=perc_loop

line += ('par', do(5,24), 'ti', do(5,24), 'dant', do(5,48), 'stap', do(5,24), la(4,24), 'nO', la(4,48))
seq_g += SoTiRe(36)+SoTiRe(36)+LaDoMi(48)+LaDoMi(24)+LaDoMi(48)
seq_p +=perc_loop

line += ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,48))
seq += [line, BL(48)]
seq_g += ReFaLaDo(36)+ReFaLaDo(36)+ReFaLaDo(24)+[BL(72)]+ReFaLaDo(24)
seq_p +=perc_loop

line = ('O', la(4,36), 'O', la(4,36), 'O', la(4,24), 'O', mi(5,48))
seq += [line, BL(48)]
seq_g +=LaDoMi(36)+LaDoMi(36)+SoTiRe(48)+SoTiRe(36)+SoTiRe(36)
seq_p +=[dong(),cha(), cha(12), cha(12),cha(12),cha(12), cha()]

doc.sing(seq, singer,track)
doc.playNoteSeq(seq_g, Guitar, track_g)
doc.playBeatSeq(seq_p, perc_list, track_p)

doc.setTrackVolume(track_g, 0.5)

doc.meteor()
doc.mixDown('tiktok.wav')
```





