<script type="text/javascript" src="meteor.js"></script>

<script type="text/javascript">
    window.onload = function() {
        var restiktok = {
            canvasID : "canvastiktok",
            audioID : "audiotiktok",
            dataPath : "ouchi.meteor"
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
        <source type="audio/mpeg" src="ouchi.mp3"/>
    </audio>
</div>

# この素晴らしい世界に祝福を！2 ED/おうちに帰りたい

## Info
* The song is from Anime この素晴らしい世界に祝福を！2 produced by Studio DEEN.
* Originally sung by: 雨宮天, 高橋李依, 茅野愛衣
* Melody/Lyric by:  佐藤良成
* Video published earlier on Bilibili: [https://www.bilibili.com/video/av19520577](https://www.bilibili.com/video/av19520577)
* 呗音デフォ子 comes with the software UTAU
* 三色あやか 連続音V2.0: [https://bowlroll.net/file/69898](https://bowlroll.net/file/69898)

```python
import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.MeteorDocument()

uta= ScoreDraft.uta_UTAU()
Ayaka= ScoreDraft.Ayaka2_UTAU()
Ayaka.setLyricConverter(ScoreDraft.JPVCVConverter)

uta.tune('pan -0.5')
Ayaka.tune('pan 0.5')

# All instrument samples from https://freewavesamples.com/
guitar = ScoreDraft.JazzGuitar()
bass = ScoreDraft.Bass()
harmonica = ScoreDraft.Harmonica()

BassDrum=ScoreDraft.BassDrum()
ClosedHitHat = ScoreDraft.ClosedHitHat()

perc_list= [BassDrum, ClosedHitHat]

def dong(duration=48):
	return (0,duration)

def chi(duration=48):
	return (1,duration)

def faS(octave=5, duration=48):
	return note(octave,Freqs[6],duration)

def soS(octave=5, duration=48):
	return note(octave,Freqs[8],duration)

track=doc.newBuf()
track_g=doc.newBuf()
track_b=doc.newBuf()
track_h=doc.newBuf()
track_p=doc.newBuf()

drum_repeat = [dong(24), chi(24), dong(24), chi(24), dong(24), chi(24), dong(24), chi(24)]

seq_guitar = [ do(7, 96), ti(6,48), la(6,24), so(6,24), BK(192)]
seq_guitar += [ do(5, 96), BK(72), mi(6,24), so(6,24), so(5,24), mi(4, 96), BK(72), ti(5,24), mi(6,24), mi(5,24) ]
seq_bass = [do(4, 96), mi(3,96)]
seq_drum = []
seq_drum += drum_repeat

seq_guitar += [la(6,24), do(7,48), la(6,24), so(6,96), BK(192)]
seq_guitar += [fa(5,96), BK(48), do(6,48), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica = [BL(192*2), do(7,24), la(6,24), so(6,24), mi(6,24), re(6,24), do(6,24), mi(6,48)]
seq_guitar += [fa(5,48), do(6,48), mi(5,48), mi(4,48)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [do(6,48), do(6,24), la(5,24), do(6,96)]
seq_guitar += [re(5,48), so(4,48), do(5,96), BK(72), mi(5,12), so(5,12), do(6,48)]
seq_bass += [re(4, 48), so(3,48), do(4,96)]
seq_drum += drum_repeat

doc.sing([BL(192*4)], uta, track)

seq= [('な', so(4,24), 'に', do(5,24), 'も', do(5,24), 'い', re(5,24), 'わ', mi(5,24), 'ず', so(5,24)), BL(24)]
seq+= [('に', so(5,24), 'い', la(5,24), 'え', do(6,24), 'うぉ', do(6,24), 'で', la(5,24), 'て', so(5,48)), BL(48)]
seq+= [('こ', la(5,24), 'ん', do(6,24), 'な', do(6,24), 'と', la(5,24), 'こ', so(5,24), 'ま', so(5,24), 'で', mi(5,36)), BL(12)]
seq+= [('き', re(5,24), 'た', do(5,24), 'け', re(5,24), 'れ', mi(5,24), 'ど', re(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(96), fa(5,36), do(6,60), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), la(5,48), so(4,96), BK(72), ti(4,12), re(5,12), so(5,24), re(5,24)]
seq_bass += [ re(4, 96), so(3,96)]
seq_drum += drum_repeat

seq+= [('ひ', so(4,24), 'ぐ', do(5,24), 'れ', do(5,24), 'と', re(5,24), 'と', mi(5,24), 'も', so(5,24)), BL(24)]
seq+= [('に', so(5,24), 'な', la(5,24), 'き', do(6,24), 'む', do(6,24), 'し', la(5,24), 'が', so(5,48)), BL(48)]
seq+= [('こ', la(5,24), 'こ', do(6,24), 'ろ', do(6,24), 'ぼ', la(5,24), 'そ', so(5,24), 'い', so(5,24), 'と', mi(5,36)), BL(12)]
seq+= [('べ', re(5,24), 'そ', mi(5,24), 'うぉ', mi(5,24), 'か', re(5,24), 'く', do(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,36), do(6,60), mi(5,48), BK(48), so(5,48), do(5,48), BK(48), so(5,48), BK(48), do(6,48)  ]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,36), BK(36), la(5,36), ti(4, 60), BK(48), so(5,48), do(5,96), BK(72), mi(5,12), so(5,12), do(6,48)]
seq_bass += [re(4, 48), so(3,48), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('あ', so(4,24), 'か', do(5,24), 'く', do(5,24), 'そ', re(5,24), 'ま', mi(5,24), 'る', so(5,24)), BL(24)]
seq+= [('ま', so(5,24), 'ち', la(5,24), 'の', do(6,24), 'そ', do(6,24), 'ら', la(5,24), 'うぉ', so(5,48)), BL(48)]
seq+= [('か', la(5,24), 'ら', do(6,24), 'す', do(6,24), 'が', la(5,24), 'な', so(5,24), 'い', so(5,24), 'て', mi(5,36)), BL(12)]
seq+= [('い', re(5,24), 'き', do(5,24), 'す', re(5,24), 'ぎ', mi(5,24), 'る', re(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(96), fa(5,36), do(6,60), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), la(5,48), so(4,96), BK(72), ti(4,12), re(5,12), so(5,24), re(5,24)]
seq_bass += [ re(4, 96), so(3,96)]
seq_drum += drum_repeat

seq+= [('み', so(4,24), 'ち', do(5,24), 'に', do(5,24), 'の', re(5,24), 'び', mi(5,24), 'る', so(5,24)), BL(24)]
seq+= [('な', so(5,24), 'が', la(5,24), 'い', do(6,24), 'か', do(6,24), 'げ', la(5,24), 'が', so(5,48)), BL(48)]
seq+= [('は', la(5,24), 'や', do(6,24), 'く', do(6,24), 'か', la(5,24), 'え', so(5,24), 'ろ', so(5,24), 'と', mi(5,36)), BL(12)]
seq+= [('そ', re(5,24), 'で', mi(5,24), 'うぉ', mi(5,24), 'ひ', re(5,24), 'く', do(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,36), do(6,60), mi(5,48), BK(48), so(5,48), do(5,48), BK(48), so(5,48), BK(48), do(6,48)  ]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,36), BK(36), la(5,36), ti(4, 60), BK(48), so(5,48), do(5,96), BK(72), mi(5,12), so(5,12), do(6,48)]
seq_bass += [re(4, 48), so(3,48), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq = [('お', la(5,24), 'さ', do(6,48), 'か', la(5,24), 'な', do(6,48), 'うぉ', la(5,36)), BL(12)]
seq +=[('や', so(5,24), 'く', mi(5,24), 'に', so(5,24), 'お', la(5,24), 'い', so(5,48), BL(48)) ]

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), la(5,12), do(6,48), BK(48), mi(6,48), la(5,96), BK(72), do(6,24), mi(6,48)]
seq_bass += [fa(3,96), la(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [mi(4,48), BK(48), so(5,48), ti(5,48), do(5,96), BK(72), re(5,12), mi(5,12), so(5,24), mi(5,24)]
seq_bass += [mi(3,96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq =[('ば', la(5,24), 'ん', do(6,48), 'ご', la(5,24), 'は', do(6,24), 'ん', do(6,24), 'の', la(5,36)), BL(12)]
seq +=[('い', so(5,24), 'い', mi(5,24), 'に', re(5,24), 'お', re(5,24), 'い', re(5,48), BL(48)) ]

seq_harmonica += [BL(192)]
seq_guitar += [fa(5,96), BK(60), la(5,12), do(6,48), BK(48), mi(6,48), faS(5,96), BK(72), do(6,24), mi(6,48)]
seq_bass += [fa(3,96), faS(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [so(5,48), BK(48), ti(5,48), re(5,48), BK(48), so(5,48), so(4,24), so(5,12), re(5,12), so(4,48)]
seq_bass += [re(4,96), so(3,96)] 
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq= [('お', so(4,24), 'な', do(5,24), 'か', do(5,24), 'の', re(5,24), 'む', mi(5,24), 'し', so(5,24)), BL(24)]
seq+= [('も', so(5,24), 'な', la(5,24), 'き', do(6,24), 'だ', do(6,24), 'し', la(5,24), 'た', so(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('い', la(5,24), 'じ', do(6,24), 'うぉ', do(6,24), 'は', la(5,24), 'る', so(5,24), 'の', so(5,24), 'も', mi(5,36)), BL(12)]
seq+= [('あ', re(5,24), 'き', do(5,24), 'て', re(5,24), 'ひ', mi(5,24), 'た', re(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(96), fa(5,36), do(6,60), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), la(5,48), so(4,96), BK(72), ti(4,12), re(5,12), so(5,24), re(5,24)]
seq_bass += [ re(4, 96), so(3,96)]
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq= [('い', so(4,24), 'ま', do(5,24), 'す', do(5,24), 'ぐ', re(5,24), 'ご', mi(5,24), 'め', so(5,12), 'ん', so(5,12)), BL(24)]
seq+= [('と', so(5,24), 'あ', la(5,24), 'や', do(6,24), 'ま', do(6,24), 'あ', re(6,24), 'て', mi(6,36)), BL(12), ('あ', mi(6,24), re(6,24))]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,12), la(6,48), mi(4,48), BK(48), mi(6,48), BK(48), soS(6,48), ti(4,48), BK(48), re(6,48), BK(48), so(6,48)]
seq_bass += [fa(3, 96), mi(3,48), re(3,48)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('は', do(6,24), 'や', la(5,24), 'く', do(6,24), 'お', la(5,24), 'う', so(5,24), 'ち', mi(5,24), 'に', do(6,36)), BL(12)]
seq+= [('か', re(5,24), 'え', mi(5,24), 'り', mi(5,24), 'た', re(5,24), 'い', do(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5, 96), BK(60), do(6,12), mi(6,48), mi(5,48), BK(48), ti(5,48), BK(48), re(6,48), la(4,48), BK(48), so(5,48), BK(48), do(6,48), BK(48), mi(6,48)]
seq_bass += [fa(3, 96), mi(3,48), la(3,48)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), BK(48), la(5,48), so(4,48), do(5,96), BK(72), mi(5,24), do(5,48), BK(48), do(6,48)]
seq_bass += [re(3,48), so(3,48), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [do(7,24), la(6,24), so(6,24), mi(6,24), re(6,24), do(6,24), mi(6,48)]
seq_guitar += [ fa(5,48), do(6,48), mi(5,48), mi(4,48)]
seq_bass += [fa(3, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [do(6,48), do(6,24), la(5,24), do(6,96)]
seq_guitar += [re(5,48), so(4,48), do(5,96), BK(96), mi(5,96), BK(96), so(5,96)]
seq_bass += [so(3, 96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq+[BK(192*2)], Ayaka, track)
doc.sing(seq+[BL(192*2)], uta, track)

seq= [('い', so(4,24), 'く', do(5,24), 'あ', do(5,24), 'て', re(5,24), 'の', mi(5,24), 'な', so(5,12), 'い', so(5,12)), BL(24)]
seq+= [('ぼ', so(5,24), 'く', la(5,24), 'の', do(6,24), 'ま', do(6,24), 'え', la(5,24), 'うぉ', so(5,48)), BL(48)]
seq+= [('こ', la(5,24), 'ど', do(6,24), 'も', do(6,24), 'が', la(5,24), 'ひ', so(5,24), 'と', so(5,24), 'り', mi(5,36)), BL(12)]
seq+= [('い', re(5,24), 'き', do(5,24), 'す', re(5,24), 'ぎ', mi(5,24), 'る', re(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(96), fa(5,36), do(6,60), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), la(5,48), so(4,96), BK(72), ti(4,12), re(5,12), so(5,24), re(5,24)]
seq_bass += [ re(4, 96), so(3,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('は', so(4,24), 'な', do(5,24), 'うぉ', do(5,24), 'す', re(5,24), 'す', mi(5,24), 'り', so(5,24)), BL(24)]
seq+= [('しゃ', so(5,24), 'く', la(5,24), 'り', do(6,24), 'あ', do(6,24), 'げ', la(5,24), 'て', so(5,48)), BL(48)]
seq+= [('わ', la(5,24), 'き', do(6,24), 'め', do(6,24), 'も', la(5,24), 'ふ', so(5,24), 'ら', so(5,24), 'ず', mi(5,36)), BL(12)]
seq+= [('は', re(5,24), 'し', mi(5,24), 'い', mi(5,24), 'て', re(5,24), 'く', do(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,36), do(6,60), mi(5,48), BK(48), so(5,48), do(5,48), BK(48), so(5,48), BK(48), do(6,48)  ]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,36), BK(36), la(5,36), ti(4, 60), BK(48), so(5,48), do(5,96), BK(72), mi(5,12), so(5,12), do(6,48)]
seq_bass += [re(4, 48), so(3,48), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq = [('や', la(5,24), 'み', do(6,48), 'に', la(5,24), 'き', do(6,48), 'え', la(5,36)), BL(12)]
seq +=[('て', so(5,24), 'く', mi(5,24), 'せ', so(5,24), 'な', la(5,24), 'か', so(5,48), BL(48)) ]

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), la(5,12), do(6,48), BK(48), mi(6,48), la(5,96), BK(72), do(6,24), mi(6,48)]
seq_bass += [fa(3,96), la(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [mi(4,48), BK(48), so(5,48), ti(5,48), do(5,96), BK(72), re(5,12), mi(5,12), so(5,24), mi(5,24)]
seq_bass += [mi(3,96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

line=('あ', la(5,24), 'の', do(6,48), 'ひ', la(5,24), 'の', do(6,48), 'ぼ', la(5,24), 'く', do(6,24))
line+=('に', re(6,24), 'に', do(6,24), 'て', re(6,24), 'い', mi(6,24), 'る', re(6,48), BL(48))
seq =[line]

seq_harmonica += [BL(192)]
seq_guitar += [fa(5,96), BK(60), la(5,12), do(6,48), BK(48), mi(6,48), faS(5,96), BK(72), do(6,24), mi(6,48)]
seq_bass += [fa(3,96), faS(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [so(5,48), BK(48), ti(5,48), re(5,48), BK(48), so(5,48), so(4,24), so(5,12), re(5,12), so(4,48)]
seq_bass += [re(4,96), so(3,96)] 
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq= [('は', so(4,24), 'し', do(5,24), 'れ', do(5,24), 'は', re(5,24), 'し', mi(5,24), 'れ', so(5,24)), BL(24)]
seq+= [('な', so(5,24), 'み', la(5,24), 'だ', do(6,24), 'ふ', do(6,24), 'い', la(5,24), 'て', so(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('か', la(5,24), 'て', do(6,24), 'た', do(6,24), 'お', la(5,24), 'つ', so(5,24), 'き', so(5,24), 'さ', mi(5,18), 'ん', mi(5,18)), BL(12)]
seq+= [('お', re(5,24), 'い', do(5,24), 'か', re(5,24), 'け', mi(5,24), 'て', re(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(96), fa(5,36), do(6,60), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), la(5,48), so(4,96), BK(72), ti(4,12), re(5,12), so(5,24), re(5,24)]
seq_bass += [ re(4, 96), so(3,96)]
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq= [('い', so(4,24), 'ま', do(5,24), 'す', do(5,24), 'ぐ', re(5,24), 'ご', mi(5,24), 'め', so(5,12), 'ん', so(5,12)), BL(24)]
seq+= [('と', so(5,24), 'あ', la(5,24), 'や', do(6,24), 'ま', do(6,24), 'れ', re(6,24), 'ば', mi(6,36)), BL(12), ('あ', mi(6,24), re(6,24))]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,12), la(6,48), mi(4,48), BK(48), mi(6,48), BK(48), soS(6,48), ti(4,48), BK(48), re(6,48), BK(48), so(6,48)]
seq_bass += [fa(3, 96), mi(3,48), re(3,48)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('ば', do(6,24), 'ん', la(5,24), 'ご', do(6,24), 'は', la(5,24), 'ん', so(5,24), 'に', mi(5,24), 'わ', do(6,36)), BL(12)]
seq+= [('ま', re(5,24), 'に', mi(5,24), 'あ', mi(5,24), 'う', re(5,24), 'さ', do(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5, 96), BK(60), do(6,12), mi(6,48), mi(5,48), BK(48), ti(5,48), BK(48), re(6,48), la(4,48), BK(48), so(5,48), BK(48), do(6,48), BK(48), mi(6,48)]
seq_bass += [fa(3, 96), mi(3,48), la(3,48)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), BK(48), la(5,48), so(4,48), do(5,96), BK(72), mi(5,24), do(5,48), BK(48), do(6,48)]
seq_bass += [re(3,48), so(3,48), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [do(7,24), la(6,24), so(6,24), mi(6,24), re(6,24), do(6,24), mi(6,48),BK(192)]
seq_guitar += [ fa(5,48), do(6,48), mi(5,48), mi(4,48)]
seq_bass += [fa(3, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [do(6,48), do(6,24), la(5,24), do(6,96),BK(192)]
seq_guitar += [re(5,48), so(4,48), do(5,96), BK(96), mi(5,96), BK(96), so(5,96)]
seq_bass += [so(3, 96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq+[BK(192*2)], Ayaka, track)
doc.sing(seq+[BL(192*2)], uta, track)

seq = [('お', la(5,24), 'さ', do(6,48), 'か', la(5,24), 'な', do(6,48), 'うぉ', la(5,36)), BL(12)]
seq +=[('や', so(5,24), 'く', mi(5,24), 'に', so(5,24), 'お', la(5,24), 'い', so(5,48), BL(48)) ]

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), la(5,12), do(6,48), BK(48), mi(6,48), la(5,96), BK(72), do(6,24), mi(6,48)]
seq_bass += [fa(3,96), la(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [mi(4,48), BK(48), so(5,48), ti(5,48), do(5,96), BK(72), re(5,12), mi(5,12), so(5,24), mi(5,24)]
seq_bass += [mi(3,96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq =[('ば', la(5,24), 'ん', do(6,48), 'ご', la(5,24), 'は', do(6,24), 'ん', do(6,24), 'の', la(5,36)), BL(12)]
seq +=[('い', so(5,24), 'い', mi(5,24), 'に', re(5,24), 'お', re(5,24), 'い', re(5,48), BL(48)) ]

seq_harmonica += [BL(192)]
seq_guitar += [fa(5,96), BK(60), la(5,12), do(6,48), BK(48), mi(6,48), faS(5,96), BK(72), do(6,24), mi(6,48)]
seq_bass += [fa(3,96), faS(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [so(5,48), BK(48), ti(5,48), re(5,48), BK(48), so(5,48), so(4,24), so(5,12), re(5,12), so(4,48)]
seq_bass += [re(4,96), so(3,96)] 
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq= [('お', so(4,24), 'な', do(5,24), 'か', do(5,24), 'の', re(5,24), 'む', mi(5,24), 'し', so(5,24)), BL(24)]
seq+= [('も', so(5,24), 'な', la(5,24), 'き', do(6,24), 'だ', do(6,24), 'し', la(5,24), 'た', so(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,60), do(5,96), BK(72), so(5,12), do(6,12), mi(6,24), so(5,24)]
seq_bass += [fa(3, 96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('い', la(5,24), 'じ', do(6,24), 'うぉ', do(6,24), 'は', la(5,24), 'る', so(5,24), 'の', so(5,24), 'も', mi(5,36)), BL(12)]
seq+= [('あ', re(5,24), 'き', do(5,24), 'て', re(5,24), 'ひ', mi(5,24), 'た', re(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(96), fa(5,36), do(6,60), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), la(5,48), so(4,96), BK(72), ti(4,12), re(5,12), so(5,24), re(5,24)]
seq_bass += [ re(4, 96), so(3,96)]
seq_drum += drum_repeat

doc.sing(seq, Ayaka, track)

seq= [('い', so(4,24), 'ま', do(5,24), 'す', do(5,24), 'ぐ', re(5,24), 'ご', mi(5,24), 'め', so(5,12), 'ん', so(5,12)), BL(24)]
seq+= [('と', so(5,24), 'あ', la(5,24), 'や', do(6,24), 'ま', do(6,24), 'あ', re(6,24), 'て', mi(6,36)), BL(12), ('あ', mi(6,24), re(6,24))]

seq_harmonica += [BL(192)]
seq_guitar += [ do(5, 96), BK(60), mi(5,12), so(5,48), mi(4,96), BK(60), mi(5,12), so(5,48), BK(48), ti(5,48)]
seq_bass += [do(4, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5,96), BK(60), do(6,12), la(6,48), mi(4,48), BK(48), mi(6,48), BK(48), soS(6,48), ti(4,48), BK(48), re(6,48), BK(48), so(6,48)]
seq_bass += [fa(3, 96), mi(3,48), re(3,48)]
seq_drum += drum_repeat

doc.sing(seq, uta, track)

seq= [('は', do(6,24), 'や', la(5,24), 'く', do(6,24), 'お', la(5,24), 'う', so(5,24), 'ち', mi(5,24), 'に', do(6,36)), BL(12)]
seq+= [('か', re(5,24), 'え', mi(5,24), 'り', mi(5,24), 'た', re(5,24), 'い', do(5,48)), BL(48)]

seq_harmonica += [BL(192)]
seq_guitar += [ fa(5, 96), BK(60), do(6,12), mi(6,48), mi(5,48), BK(48), ti(5,48), BK(48), re(6,48), la(4,48), BK(48), so(5,48), BK(48), do(6,48), BK(48), mi(6,48)]
seq_bass += [fa(3, 96), mi(3,48), la(3,48)]
seq_drum += drum_repeat

seq_harmonica += [BL(192)]
seq_guitar += [ re(5,48), BK(48), la(5,48), so(4,48), do(5,96), BK(72), mi(5,24), do(5,48), BK(48), do(6,48)]
seq_bass += [re(3,48), so(3,48), do(4,96)]
seq_drum += drum_repeat

seq_harmonica += [do(7,24), la(6,24), so(6,24), mi(6,24), re(6,24), do(6,24), mi(6,48)]
seq_guitar += [ fa(5,48), do(6,48), mi(5,48), mi(4,48)]
seq_bass += [fa(3, 96), mi(3,96)]
seq_drum += drum_repeat

seq_harmonica += [do(6,48), do(6,24), la(5,24), do(6,96)]
seq_guitar += [re(5,48), so(4,48), do(5,96), BK(96), mi(5,96), BK(96), so(5,96)]
seq_bass += [so(3, 96), do(4,96)]
seq_drum += drum_repeat

doc.sing(seq+[BK(192*2)], Ayaka, track)
doc.sing(seq+[BL(192*2)], uta, track)

doc.playNoteSeq(seq_harmonica, harmonica, track_h)
doc.playNoteSeq(seq_guitar, guitar, track_g)
doc.playNoteSeq(seq_bass, bass, track_b)
doc.playBeatSeq(seq_drum, perc_list, track_p)

doc.setTrackVolume(track_h, 0.5)
doc.setTrackVolume(track_g, 0.5)
doc.setTrackVolume(track_b, 0.5)
doc.setTrackVolume(track_p, 0.5)

doc.setTrackPan(track_h, 0.2)
doc.setTrackPan(track_g, -0.2)
doc.setTrackPan(track_b, 0.2)
doc.setTrackPan(track_p, -0.2)

doc.meteor()
doc.mixDown('ouchi ni kaeritai.wav')
```
