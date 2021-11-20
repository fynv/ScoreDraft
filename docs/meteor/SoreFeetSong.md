<script type="text/javascript" src="meteor.js"></script>

<script type="text/javascript">
    window.onload = function() {
        var restiktok = {
            canvasID : "canvastiktok",
            audioID : "audiotiktok",
            dataPath : "SoreFeetSong.meteor"
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
        <source type="audio/mpeg" src="SoreFeetSong.mp3"/>
    </audio>
</div>

# The Sore Feet Song

## Info
* The song is from Anime 蟲師 produced by ARTLAND.
* Originally sung by: Ally Kerr
* Melody/Lyric by: Ally Kerr
* WALTTenglish Z VCCV: http://utau.wikia.com/wiki/WALTT

```python
import ScoreDraft
from ScoreDraft.Notes import *

def miF(octave=5, duration=48):
	return note(octave,Freqs[3],duration)

doc=ScoreDraft.MeteorDocument()

singer = ScoreDraft.WALTT_Z_UTAU()
singer.setLyricConverter(ScoreDraft.VCCVEnglishConverter)
singer.setCZMode()

Guitar = ScoreDraft.NylonGuitar()

FreqsB=[f*Freqs[11]*0.5 for f in Freqs]
Freqs[:]=FreqsB

track=doc.newBuf()
track_g=doc.newBuf()

def Repeat(x,t):
	ret=[]
	for i in range(t):
		ret+=x
	return ret


seq=[BL(192)]
seq_g = [so(4,32),re(5,16),so(5,96),BK(96),re(6,32),ti(5,16),do(6,32),ti(5,16),re(5,48),BK(48),so(5,48)]

seq+=[BL(192)]
seq_g += [fa(4,32),do(5,16),fa(5,96), BK(144), la(5,32), so(5,48), la(5,48), so(5,64), BK(48), do(5,48)]

seq+=[BL(192)]
seq_g +=  [do(4,32),so(4,16),do(5,96),BK(96),re(6,32),ti(5,16),do(6,32),ti(5,16),so(4,48),BK(48),so(5,48)]

seq+=[BL(192)]
seq_g += [so(4,32),re(5,16),so(5,96), BK(144), la(5,32), so(5,48), ti(5,48), so(5,64), BK(48), re(4,48)]

seq+=[BL(48)]
line= ('I',so(4,48),'w9lk',so(4,48),'ten',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48)]

line +=('th8',fa(4,24),'zxnd',so(4,48),'mIls',so(4,72),'ten',so(4,48))
seq_g += [fa(3,32),do(4,16),fa(4,96), do(4,48)]

line +=('th8',mi(4,24),'zxnd',so(4,48),'mIls',so(4,72),'t6',so(4,24))
seq_g += [do(4,32),so(4,16),do(5,96), do(4,48)]

line +=('sE',ti(4,48),'yx',so(4,48))
seq+=[line, BL(120)]
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144), re(6,32), ti(5,16), do(6,32), ti(5,16), so(5,48)]

seq+=[BL(48)]
line= ('&amp;nd',so(4,48),'ev',so(4,48),'ri',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48)]

line +=('g@sp',fa(4,24),'xv',so(4,48),'breth',so(4,72),'I',so(4,48))
seq_g += [fa(3,32),do(4,16),fa(4,96), do(4,48)]

line +=('gr@bd',mi(4,24),'it',so(4,48),'zhust',so(4,72),'t6',so(4,24))
seq_g += [do(4,32),so(4,16),do(5,96), do(4,48)]

line +=('fInd',ti(4,48),'yx',so(4,48))
seq+=[line, BL(120)]
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144), re(6,32), ti(5,16), do(6,32), ti(5,16), so(5,48)]

seq+=[BL(48)]
line= ('I',so(4,48),'klImbd',so(4,48),'up',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144),re(5,48),re(5,48),re(5,48)]

line+=('ev',do(5,48),'ri',ti(4,48),'hill',la(4,48),'t6',ti(4,32))
seq_g+=[fa(3,48),do(4,48),fa(4,48),do(4,48), BK(192), so(5,48), so(5,48),so(5,48),so(5,32)]

line+=('get',so(4,112))
seq+=[line, BL(32)]
line = ('t6', so(4,24), la(4,40))
seq_g+=[mi(5,112), BL(32), mi(5,64), BK(192), do(3,32),so(3,16),do(4,96),so(3,48)]

line +=('yx', so(4,96))
seq+=[line, BL(96)]
seq_g+=[mi(5,96), BK(96), so(3,32),re(4,16),so(4,96), re(4,48), BK(144), re(6,32), ti(5,16), do(6,32), ti(5,16), so(5,48)]

seq+=[BL(48)]
line= ('I',so(4,48),'wan',so(4,48),'d3d',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144),re(5,48),re(5,48),re(5,48)]

line+=('An',do(5,48),'shent',ti(4,48),'l&amp;nd',la(4,48),'st6',ti(4,32))
seq_g+=[fa(3,48),do(4,48),fa(4,48),do(4,48), BK(192), so(5,48), so(5,48),so(5,48),so(5,32)]

line+=('hOd',so(4,112))
seq+=[line, BL(32)]
line = ('zhust', so(4,24), la(4,40))
seq_g+=[mi(5,112), BL(32), mi(5,64), BK(192), do(3,32),so(3,16),do(4,96),so(3,48)]

line +=('yx', so(4,96))
seq+=[line, BL(96)]
seq_g+=[so(3,32),re(4,16),so(4,32),so(4,48),so(4,16), so(4,12),re(4,12),so(3,24), BK(192), mi(5,192)]

seq+=[BL(48)]
line= ('&amp;nd',fa(4,48),'ev',fa(4,48),'ri',fa(4,48))
seq_g+=[re(3,32),la(3,16),re(4,96),la(3,48)]

line+=('s1ng', fa(4,48), 'gx', fa(4,32), 'step',mi(4,48), 'xv', do(4,16), re(4,12),'dhx', do(4,36))
seq_g+=[fa(3,32),do(4,16),fa(4,96),fa(3,48)]

line+=('wA',do(4,96), BL(72), 'I',do(4,24))
seq_g+=[do(4,32),so(4,16),do(5,96),BK(96), do(5,32), do(5,48), do(5,48), do(5,16)]

line+=('pA',do(4,48),'A', mi(4,48),'A',fa(4,48),'A',mi(4,48))
seq_g+=Repeat([do(4,48),BK(42),so(4,42),BK(36),do(5,36)],4)

line+=('Ad',re(4,72))
seq+=[line,BL(24)]
line=('ev',fa(4,48),'ri',fa(4,48))
seq_g+=[re(3,24),la(3,24),re(4,48),BL(48),re(4,48)]

line+=('s1ng', fa(4,48), 'gxl', so(4,48), 'nIt', la(4,48), '&amp;n', ti(4,24),do(5,168))
seq+=[line, BL(48)]
seq_g+=[so(3,48),fa(4,48),la(4,48),do(5,48), BL(192)]

line=('dA',ti(4,144), 'A',la(4,48),so(4,144), 'I',so(4,24),la(4,24))
seq_g+=[so(3,192), BK(192), so(4,24),re(4,24),so(3,24),so(4,48),so(3,24),so(4,48)]
seq_g+=[so(3,192), BK(192), fa(4,24),re(4,24),so(3,24),fa(4,48),so(3,24),fa(4,48)]

line+=('s3ch',so(4,144),'f0r',fa(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), BL(48)]

line+=('yx', so(4,120))
seq+=[line, BL(24)]
line= ('thro',re(4,48))
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), BL(48)]

line+=('s&amp;nd', ti(4,72),'st0rms', ti(4,72), '&amp;nd', do(5,48))
seq_g+=[so(3,96), BK(96), so(4,24), re(4,24), so(3,24), so(4,24), re(4,24), so(3,24), so(4,24),re(4,24), BK(192), so(5,72), so(5,72), so(5,48)] 

line+=('hA', ti(4,48), 'zi', la(4,24), 'dans', so(4,60))
seq+=[line, BL(12)]
line=('I', so(4,24),la(4,24))
seq_g+=[so(5,48), BK(48), so(3,96), BK(96), fa(4,24), re(4,24), so(3,24), fa(4,24), re(4,24), so(3,24), fa(4,24),re(4,24)] 

line+=('rEch',so(4,144),'f0r',fa(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), so(4,24), BL(24)]

line+=('yx', so(4,144), la(4,48), so(4,144))
seq+=[line, BL(48)]
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), so(4,24), BL(24)]
seq_g+=[do(4,192), BK(168), re(4,24), so(4,24),do(5,24),so(4,24),re(4,24), so(3,48), BK(42), do(4,42), BK(36), re(4,36), BK(30), so(4,30) ]

seq+=[BL(48)]
line= ('I',so(4,48),'st0l',so(4,48),'ten',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48)]

line +=('th8',fa(4,24),'zxnd',so(4,48),'p9nd',so(4,72),'ten',so(4,48))
seq_g += [fa(3,32),do(4,16),fa(4,96), do(4,48)]

line +=('th8',mi(4,24),'zxnd',so(4,48),'p9nd',so(4,72),'t6',so(4,24))
seq_g += [do(4,32),so(4,16),do(5,96), do(4,48)]

line +=('sE',ti(4,48),'yx',so(4,48))
seq+=[line, BL(120)]
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144), re(6,32), ti(5,16), do(6,32), ti(5,16), so(5,48)]

seq+=[BL(48)]
line= ('I',so(4,48),'rubd',so(4,48),'kxn',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48)]

line +=('vE',fa(4,24),'nixn',so(4,48),'st0r',so(4,72),'kas',so(4,24),'I',so(4,24))
seq_g += [fa(3,32),do(4,16),fa(4,96), do(4,48)]

line +=('th8t',mi(4,24),'dhAd',so(4,48),'mAk',so(4,72),'it',so(4,24))
seq_g += [do(4,32),so(4,16),do(5,96), do(4,48)]

line +=('E',ti(4,48),'zEr',so(4,48))
seq+=[line, BL(120)]
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144), re(6,32), ti(5,16), do(6,32), ti(5,16), so(5,48)]

seq+=[BL(48)]
line= ('I',so(4,48),'livd',so(4,48),'9f',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144),re(5,48),re(5,48),re(5,48)]

line+=('r@ts',do(5,48),'&amp;nd',ti(4,48),'tOds',la(4,32),'&amp;nd', la(4,24), 'I', ti(4,24))
seq_g+=[fa(3,48),do(4,48),fa(4,48),do(4,48), BK(192), so(5,48), so(5,48),so(5,48),so(5,32)]

line+=('starv',so(4,112))
seq+=[line, BL(32)]
line = ('f0r', so(4,24), la(4,40))
seq_g+=[mi(5,112), BL(32), mi(5,64), BK(192), do(3,32),so(3,16),do(4,96),so(3,48)]

line +=('yx', so(4,96))
seq+=[line, BL(96)]
seq_g+=[mi(5,96), BK(96), so(3,32),re(4,16),so(4,96), re(4,48), BK(144), re(6,32), ti(5,16), do(6,32), ti(5,16), so(5,48)]

seq+=[BL(48)]
line= ('I',so(4,48),'f8t',so(4,48),'9f',so(4,48))
seq_g += [so(3,32),re(4,16),so(4,96), re(4,48), BK(144),re(5,48),re(5,48),re(5,48)]

line+=('zhI',do(5,48),'xnt',ti(4,48),'bAr',la(4,32),'&amp;nd',la(4,24),'I',ti(4,24))
seq_g+=[fa(3,48),do(4,48),fa(4,48),do(4,48), BK(192), so(5,48), so(5,48),so(5,48),so(5,32)]

line+=('kild',so(4,112))
seq+=[line, BL(32)]
line = ('dhem', so(4,24), la(4,40))
seq_g+=[mi(5,112), BL(32), mi(5,64), BK(192), do(3,32),so(3,16),do(4,96),so(3,48)]

line +=('to', so(4,96))
seq+=[line, BL(96)]
seq_g+=[so(3,32),re(4,16),so(4,32),so(4,48),so(4,16), so(4,12),re(4,12),so(3,24), BK(192), mi(5,192)]

seq+=[BL(48)]
line= ('&amp;nd',fa(4,48),'ev',fa(4,48),'ri',fa(4,48))
seq_g+=[re(3,32),la(3,16),re(4,96),la(3,48)]

line+=('s1ng', fa(4,48), 'gx', fa(4,32), 'step',mi(4,48), 'xv', do(4,16), re(4,12),'dhx', do(4,36))
seq_g+=[fa(3,32),do(4,16),fa(4,96),fa(3,48)]

line+=('wA',do(4,96), BL(72), 'I',do(4,24))
seq_g+=[do(4,32),so(4,16),do(5,96),BK(96), do(5,32), do(5,48), do(5,48), do(5,16)]

line+=('pA',do(4,48),'A', mi(4,48),'A',fa(4,48),'A',mi(4,48))
seq_g+=Repeat([do(4,48),BK(42),so(4,42),BK(36),do(5,36)],4)

line+=('Ad',re(4,72))
seq+=[line,BL(24)]
line=('ev',fa(4,48),'ri',fa(4,48))
seq_g+=[re(3,24),la(3,24),re(4,48),BL(48),re(4,48)]

line+=('s1ng', fa(4,48), 'gxl', so(4,48), 'nIt', la(4,48), '&amp;n', ti(4,24),do(5,168))
seq+=[line, BL(48)]
seq_g+=[so(3,48),fa(4,48),la(4,48),do(5,48), BL(192)]

line=('dA',ti(4,144), 'A',la(4,48),so(4,144), 'I',so(4,24),la(4,24))
seq_g+=[so(3,192), BK(192), so(4,24),re(4,24),so(3,24),so(4,48),so(3,24),so(4,48)]
seq_g+=[so(3,192), BK(192), fa(4,24),re(4,24),so(3,24),fa(4,48),so(3,24),fa(4,48)]

line+=('s3ch',so(4,144),'f0r',fa(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), BL(48)]

line+=('yx', so(4,120))
seq+=[line, BL(24)]
line= ('thro',re(4,48))
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), BL(48)]

line+=('s&amp;nd', ti(4,72),'st0rms', ti(4,72), '&amp;nd', do(5,48))
seq_g+=[so(3,96), BK(96), so(4,24), re(4,24), so(3,24), so(4,24), re(4,24), so(3,24), so(4,24),re(4,24), BK(192), so(5,72), so(5,72), so(5,48)] 

line+=('hA', ti(4,48), 'zi', la(4,24), 'dans', so(4,60))
seq+=[line, BL(12)]
line=('I', so(4,24),la(4,24))
seq_g+=[so(5,48), BK(48), so(3,96), BK(96), fa(4,24), re(4,24), so(3,24), fa(4,24), re(4,24), so(3,24), fa(4,24),re(4,24)] 

line+=('rEch',so(4,144),'f0r',fa(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), so(4,24), BL(24)]

line+=('yx', so(4,120))
seq+=[line, BL(24)]
line=('Im', ti(4,24),do(5,24))
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), so(4,24), BL(24)]

line+=('tIrd', ti(4,72),'&amp;nd', ti(4,72), 'Im', do(5,48))
seq_g+=[so(3,96), BK(96), so(4,24), re(4,24), so(3,24), so(4,24), re(4,24), so(3,24), so(4,24),re(4,24), BK(192), so(5,72), so(5,72), so(5,48)] 

line+=('wEk', re(5,72), 'but', do(5,60))
seq+=[line, BL(12)]
line=('Im', ti(4,48))
seq_g+=[so(5,48), BK(48), so(3,96), BK(96), fa(4,24), re(4,24), so(3,24), fa(4,24), re(4,24), so(3,24), fa(4,24),re(4,24)] 

line+=('strang',so(4,144),'f0r',so(4,24),la(4,24))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), so(4,24), BL(24)]

line+=('yx', so(4,120))
seq+=[line, BL(24)]
line=('I', ti(4,24),do(5,24))
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), so(4,24), BL(24)]

line+=('wan', ti(4,72),'t6', ti(4,72), 'gO', do(5,48))
seq_g+=[so(3,96), BK(96), so(4,24), re(4,24), so(3,24), so(4,24), re(4,24), so(3,24), so(4,24),re(4,24), BK(192), so(5,72), so(5,72), so(5,48)] 

line+=('hOm', re(5,72), 'but', do(5,60))
seq+=[line, BL(12)]
line=('mI', ti(4,48))
seq_g+=[so(5,48), BK(48), so(3,96), BK(96), fa(4,24), re(4,24), so(3,24), fa(4,24), re(4,24), so(3,24), fa(4,24),re(4,24)] 

line+=('luv',so(4,96),'gets',so(4,48),'mi',la(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), so(4,24), BL(24)]

line+=('thro', so(4,120))
seq+=[line, BL(24)]
line=('la', ti(4,24),do(5,24))
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), so(4,24), BL(24)]

line+=('la', ti(4,72),'la', ti(4,72), 'la', do(5,48))
seq_g+=[so(3,96), BK(96), so(4,24), re(4,24), so(3,24), so(4,24), re(4,24), so(3,24), so(4,24),re(4,24), BK(192), so(5,72), so(5,72), so(5,48)] 

line+=('la', re(5,72), 'la', do(5,60))
seq+=[line, BL(12)]
line=('la', ti(4,48))
seq_g+=[so(5,48), BK(48), so(3,96), BK(96), fa(4,24), re(4,24), so(3,24), fa(4,24), re(4,24), so(3,24), fa(4,24),re(4,24)] 

line+=('la',so(4,96),'la',so(4,48),'la',la(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), so(4,24), BL(24)]

line+=('la', so(4,120))
seq+=[line, BL(24)]
line=('la', ti(4,24),do(5,24))
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), so(4,24), BL(24)]

line+=('la', ti(4,72),'la', ti(4,72), 'la', do(5,48))
seq_g+=[so(3,96), BK(96), so(4,24), re(4,24), so(3,24), so(4,24), re(4,24), so(3,24), so(4,24),re(4,24), BK(192), so(5,72), so(5,72), so(5,48)] 

line+=('la', re(5,72), 'la', do(5,60))
seq+=[line, BL(12)]
line=('la', ti(4,48))
seq_g+=[so(5,48), BK(48), so(3,96), BK(96), fa(4,24), re(4,24), so(3,24), fa(4,24), re(4,24), so(3,24), fa(4,24),re(4,24)] 

line+=('la',so(4,96),'la',so(4,48),'la',la(4,48))
seq_g+=[do(4,192), BK(168), mi(4,24), so(4,24),do(5,24),so(4,24),mi(4,24), so(4,24), BL(24)]

line+=('la', so(4,144))
seq+=[line, BL(24)]
seq_g+=[do(4,192), BK(168), miF(4,24), so(4,24),do(5,24),so(4,24),miF(4,24), so(3,96), BK(90), do(4,90), BK(84), re(4,84), BK(78), so(4,78)]


#seq+=[line, BL(48)]

#print(ScoreDraft.TellDuration(seq))
#print(ScoreDraft.TellDuration(seq_g))

doc.setTempo(140)
doc.sing(seq, singer, track)
doc.playNoteSeq(seq_g,Guitar, track_g)

#doc.setTrackVolume(track_g, 0.5)

doc.meteor()
doc.mixDown('SoreFeetSong.wav')
```
