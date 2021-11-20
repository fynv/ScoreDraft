<script type="text/javascript" src="meteor.js"></script>

<script type="text/javascript">
    window.onload = function() {
        var restiktok = {
            canvasID : "canvastiktok",
            audioID : "audiotiktok",
            dataPath : "TaLang.meteor"
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
        <source type="audio/mpeg" src="TaLang.mp3"/>
    </audio>
</div>

# 踏浪

## Info
* Originally sung by: 沈雁
* Melody by: 庄奴
* Lyric by: 古月
* Video published earlier on Bilibili: [https://www.bilibili.com/video/av19520577](https://www.bilibili.com/video/av19520577)
* 趙英子 VoiceBank [http://tsuro.lofter.com/choueiko](http://tsuro.lofter.com/choueiko)

```python
import ScoreDraft
from ScoreDraft.Notes import *

doc=ScoreDraft.MeteorDocument()
doc.setTempo(72)

ZhaoYingzi = ScoreDraft.ZhaoYingzi_UTAU()
ZhaoYingzi.setLyricConverter(ScoreDraft.CVVCChineseConverter)

Guitar=ScoreDraft.NylonGuitar()
Bass=ScoreDraft.Bass()
Ocarina=ScoreDraft.Ocarina()

BassDrum=ScoreDraft.BassDrum()
Snare=ScoreDraft.Snare()
ClosedHitHat=ScoreDraft.ClosedHitHat()
Tom1=ScoreDraft.Tom1()
Tom2=ScoreDraft.Tom2()
Tom3=ScoreDraft.Tom3()
Tom4=ScoreDraft.Tom4()

perc_list=[BassDrum,Snare,ClosedHitHat,Tom1,Tom2,Tom3,Tom4]

def dong(duration=48):
	return (0,duration)

def cha(duration=48):
	return (1,duration)

def chi(duration=48):
	return (2,duration)

def tong1(duration=48):
	return (3,duration)

def tong2(duration=48):
	return (4,duration)

def tong3(duration=48):
	return (5,duration)

def tong4(duration=48):
	return (6,duration)

Seashore=ScoreDraft.Seashore()

effect_list=[Seashore]

track=doc.newBuf()
track_g=doc.newBuf()
track_b=doc.newBuf()
track_o=doc.newBuf()
track_p=doc.newBuf()
track_effect=doc.newBuf()

def faS(octave=5, duration=48):
	return note(octave,Freqs[6],duration)

def soS(octave=5, duration=48):
	return note(octave,Freqs[8],duration)

seq=[BL(192)]
seq_g=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_o=[la(5,24), la(5,12), ti(5,12), do(6,24), ti(5,24), la(5, 48), mi(5,48)]
seq_e=[(-1, 144), (0,192*2)]

seq+=[BL(192)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_o+=[la(5,24), la(5,12), ti(5,12), do(6,24), ti(5,24), la(5, 96)]

seq+=[BL(192)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_o+=[la(5,24), la(5,12), ti(5,12), do(6,24), ti(5,24), la(5, 48), mi(5,48)]
seq_e+=[(0,192*2)]

seq+=[BL(192)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), do(5,48), BK(46), mi(5,46), BK(44), la(5,44), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_o+=[la(5,24), la(5,12), ti(5,12), do(6,24), ti(5,24), la(5, 96)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]

seq+=[('xiao', la(4,24), 'xiao', la(4,12), 'di', do(5,12), 'yi', re(5,24), 'pian', mi(5,24), 'yun', re(5,12), mi(5,24),re(5,12), 'ya',re(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), fa(5,12), la(5,24), fa(5,24) ]

seq+=[('man', la(4,24), 'man', la(4,12), 'di', do(5,12), 'zou', re(5,24), 'guo', mi(5,24), 'lai', re(5,12), mi(5,72)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24)]

seq+=[('qing', la(4,24), 'ni', la(4,12), do(5,12), 'xie', re(5,24), 'xie', mi(5,24), 'jiao', re(5,12), mi(5,24),re(5,12), 'ya',re(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), faS(5,12), la(5,24), fa(5,24) ]

seq+=[('zan', mi(5,12), re(5,12), 'shi', do(5,12), re(5,12), 'ting', do(5,24), 'xia', ti(4,24), 'lai', la(4,72)), BL(24)]
seq_g+=[soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]

progress_g=ScoreDraft.TellDuration(seq_g)
progress_o=ScoreDraft.TellDuration(seq_o)

line=('shan',la(5,12), so(5,12), 'shang', mi(5,12),'di',mi(5,12), 'shan',la(5,12), so(5,12), 'hua', mi(5,24))
line+=('kai',so(5,12), la(5,24), so(5,12), 'ya', mi(5,24), so(5,12),la(5,12))
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24),la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_o+=[BL(progress_g-progress_o), la(6,48),mi(6,48)]
seq_o+=[so(6,24),mi(6,12),re(6,12),mi(6,48)]

line+=('wo', so(5,12), mi(5,12), 'cai', mi(5,12), 'dao', re(5,12), 'shan', do(5,24), 'shang', so(5,12), mi(5,12), 'lai', mi(5,72))
seq+=[line, BL(24)]
seq_g+=[so(4,96), BK(72), do(5,12), mi(5,12), so(5,24), mi(5,24), soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24)]
seq_o+=[so(6,24),mi(6,12),re(6,12),do(6,24),re(6,12),so(6,12)]
miFa=[mi(6,6),fa(6,6)]
miFaX2=miFa+miFa
miFaX4=miFaX2+miFaX2
seq_o+=miFaX4+[mi(6,48)]

seq+=[('yuan',la(4,24),'lai',la(4,12),do(5,12),'ni',re(5,24),'ye',mi(5,12),'shi',fa(5,12),'shang', re(5,12),mi(5,24),re(5,12),'shan',re(5,36)),BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), faS(5,12), la(5,24), fa(5,24) ]
seq_o+=[la(5,48), mi(6,48), la(5,12), mi(6,12), la(5,12), mi(6,12), re(6,48)]

seq+=[('kan',mi(5,12),re(5,12),'na',do(5,12),re(5,12),'shan',do(5,24),'hua',ti(4,24),'kai',la(4,72)),BL(24)]
seq_g+=[soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_o+=[do(6,24),re(6,12),mi(6,12),re(6,12),do(6,12),ti(5,24)]
laTi=[la(5,6),ti(5,6)]
laTiX2=laTi+laTi
laTiX4=laTiX2+laTiX2
seq_o+=laTiX4+[la(5,48)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]

progress=ScoreDraft.TellDuration(seq_g)
seq_b=[BL(progress-96), mi(3,48), la(3,48)]
seq_p=[BL(progress-48), tong4(12), tong3(12), tong2(12), tong1(12)]

perc_repeat=[dong(24), chi(24), cha(24), chi(24),dong(24), chi(24), cha(24), chi(24)]

seq+=[('xiao', la(4,24), 'xiao', la(4,12), 'di', do(5,12), 'yi', re(5,24), 'zhen', mi(5,24), 'feng', re(5,12), mi(5,24),re(5,12), 'ya',re(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), fa(5,12), la(5,24), fa(5,24) ]
seq_b+=[la(3,96), fa(3,24), re(3,48), fa(3,24)]
seq_p+=perc_repeat

seq+=[('man', la(4,24), 'man', la(4,12), 'di', do(5,12), 'zou', re(5,24), 'guo', mi(5,24), 'lai', re(5,12), mi(5,72)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24)]
seq_b+=[la(3,96), soS(3,24), mi(3,48), soS(3,24)]
seq_p+=perc_repeat

seq+=[('qing', la(4,24), 'ni', la(4,12), do(5,12), 'xie', re(5,24), 'xie', mi(5,24), 'jiao', re(5,12), mi(5,24),re(5,12), 'ya',re(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), faS(5,12), la(5,24), fa(5,24) ]
seq_b+=[la(3,96), faS(3,24), re(3,48), faS(3,24)]
seq_p+=perc_repeat

seq+=[('zan', mi(5,12), re(5,12), 'shi', do(5,12), re(5,12), 'ting', do(5,24), 'xia', ti(4,24), 'lai', la(4,72)), BL(24)]
seq_g+=[soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[mi(4,48), ti(3,48), la(3,96)]
seq_p+=[dong(24), chi(24), cha(24), chi(24),dong(24), chi(24), tong4(12), tong3(12), tong2(12), tong1(12)]

progress_g=ScoreDraft.TellDuration(seq_g)
progress_o=ScoreDraft.TellDuration(seq_o)

line=('hai',la(5,12), so(5,12), 'shang', mi(5,12),'di',mi(5,12), 'lang',la(5,12), so(5,12), 'hua', mi(5,24))
line+=('kai',so(5,12), la(5,24), so(5,12), 'ya', mi(5,24), so(5,12),la(5,12))
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24),la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[la(3,48), mi(3,48), la(3,48), mi(3,48)]
seq_p+=perc_repeat
seq_o+=[BL(progress_g-progress_o), la(6,48),mi(6,48)]
seq_o+=[so(6,24),mi(6,12),re(6,12),mi(6,48)]

line+=('wo', so(5,12), mi(5,12), 'cai', mi(5,12), 'dao', re(5,12), 'hai', do(5,24), 'bian', so(5,12), mi(5,12), 'lai', mi(5,72))
seq+=[line, BL(24)]
seq_g+=[so(4,96), BK(72), do(5,12), mi(5,12), so(5,24), mi(5,24), soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24)]
seq_b+=[do(4,48), so(3,48), mi(3,24), soS(3,48), mi(3,24)]
seq_p+=perc_repeat
seq_o+=[so(6,24),mi(6,12),re(6,12),do(6,24),re(6,12),so(6,12)]
seq_o+=miFaX4+[mi(6,48)]

seq+=[('yuan',la(4,24),'lai',la(4,12),do(5,12),'ni',re(5,24),'ye',mi(5,12),'ai',fa(5,12),'lang', re(5,12),mi(5,24),re(5,12),'hua',re(5,36)),BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), faS(5,12), la(5,24), fa(5,24) ]
seq_b+=[la(3,48), mi(3,48), la(3,48), re(3,48)]
seq_p+=perc_repeat
seq_o+=[la(5,48), mi(6,48), la(5,12), mi(6,12), la(5,12), mi(6,12), re(6,48)]

seq+=[('cai',mi(5,12),re(5,12),'dao',do(5,12),re(5,12),'hai',do(5,24),'bian',ti(4,24),'lai',la(4,72)),BL(24)]
seq_g+=[soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[soS(3,48), mi(3,48), la(3,24), mi(3,48), la(3,24)]
seq_p+=[dong(24), chi(24), cha(24), chi(24),dong(24), chi(24), tong4(12), tong3(12), tong2(12), tong1(12)]
seq_o+=[do(6,24),re(6,12),mi(6,12),re(6,12),do(6,12),ti(5,24)]
seq_o+=laTiX4+[la(5,48)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[la(3,48), mi(3,48), la(3,48), mi(3,48)]
seq_p+=perc_repeat

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_b+=[la(3,48), mi(3,48), la(3,96)]
seq_p+=perc_repeat

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[la(3,48), mi(3,48), la(3,48), mi(3,48)]
seq_p+=perc_repeat

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_b+=[la(3,48), mi(3,48), la(3,96)]
seq_p+=[dong(24), chi(24), cha(24), chi(24),dong(24), chi(24), tong4(12), tong3(12), tong2(12), tong1(12)]

seq+=[('xiao', la(4,24), 'xiao', la(4,12), 'di', do(5,12), 'yi', re(5,24), 'pian', mi(5,24), 'yun', re(5,12), mi(5,24),re(5,12), 'ya',re(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), fa(5,12), la(5,24), fa(5,24) ]
seq_b+=[la(3,96), fa(3,24), re(3,48), fa(3,24)]
seq_p+=perc_repeat

seq+=[('man', la(4,24), 'man', la(4,12), 'di', do(5,12), 'zou', re(5,24), 'guo', mi(5,24), 'lai', re(5,12), mi(5,72)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24)]
seq_b+=[la(3,96), soS(3,24), mi(3,48), soS(3,24)]
seq_p+=perc_repeat

seq+=[('qing', la(4,24), 'ni', la(4,12), do(5,12), 'xie', re(5,24), 'xie', mi(5,24), 'jiao', re(5,12), mi(5,24),re(5,12), 'ya',re(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), faS(5,12), la(5,24), fa(5,24) ]
seq_b+=[la(3,96), faS(3,24), re(3,48), faS(3,24)]
seq_p+=perc_repeat

seq+=[('zan', mi(5,12), re(5,12), 'shi', do(5,12), re(5,12), 'ting', do(5,24), 'xia', ti(4,24), 'lai', la(4,72)), BL(24)]
seq_g+=[soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[mi(4,48), ti(3,48), la(3,96)]
seq_p+=[dong(24), chi(24), cha(24), chi(24),dong(24), chi(24), tong4(12), tong3(12), tong2(12), tong1(12)]

progress_g=ScoreDraft.TellDuration(seq_g)
progress_o=ScoreDraft.TellDuration(seq_o)

line=('shan',la(5,12), so(5,12), 'shang', mi(5,12),'di',mi(5,12), 'shan',la(5,12), so(5,12), 'hua', mi(5,24))
line+=('kai',so(5,12), la(5,24), so(5,12), 'ya', mi(5,24), so(5,12),la(5,12))
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24),la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[la(3,48), mi(3,48), la(3,48), mi(3,48)]
seq_p+=perc_repeat
seq_o+=[BL(progress_g-progress_o), la(6,48),mi(6,48)]
seq_o+=[so(6,24),mi(6,12),re(6,12),mi(6,48)]

line+=('wo', so(5,12), mi(5,12), 'cai', mi(5,12), 'dao', re(5,12), 'shan', do(5,24), 'shang', so(5,12), mi(5,12), 'lai', mi(5,72))
seq+=[line, BL(24)]
seq_g+=[so(4,96), BK(72), do(5,12), mi(5,12), so(5,24), mi(5,24), soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24)]
seq_b+=[do(4,48), so(3,48), mi(3,24), soS(3,48), mi(3,24)]
seq_p+=perc_repeat
seq_o+=[so(6,24),mi(6,12),re(6,12),do(6,24),re(6,12),so(6,12)]
seq_o+=miFaX4+[mi(6,48)]

seq+=[('yuan',la(4,24),'lai',la(4,12),do(5,12),'ni',re(5,24),'ye',mi(5,12),'shi',fa(5,12),'shang', re(5,12),mi(5,24),re(5,12),'shan',re(5,36)),BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), re(5,12), faS(5,12), la(5,24), fa(5,24) ]
seq_b+=[la(3,48), mi(3,48), la(3,48), re(3,48)]
seq_p+=perc_repeat
seq_o+=[la(5,48), mi(6,48), la(5,12), mi(6,12), la(5,12), mi(6,12), re(6,48)]

seq+=[('kan',mi(5,12),re(5,12),'na',do(5,12),re(5,12),'shan',do(5,24),'hua',ti(4,24),'kai',la(4,72)),BL(24)]
seq_g+=[soS(4,96), BK(72), ti(4,12), mi(5,12), soS(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[soS(3,48), mi(3,48), la(3,24), mi(3,48), la(3,24)]
seq_p+=[dong(24), chi(24), cha(24), chi(24),dong(24), chi(24), tong4(12), tong3(12), tong2(12), tong1(12)]
seq_o+=[do(6,24),re(6,12),mi(6,12),re(6,12),do(6,12),ti(5,24)]
seq_o+=laTiX4+[la(5,48)]

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[la(3,48), mi(3,48), la(3,48), mi(3,48)]
seq_p+=perc_repeat

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_b+=[la(3,48), mi(3,48), la(3,96)]
seq_p+=perc_repeat

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 48), 'la', mi(5,36)), BL(12)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24)]
seq_b+=[la(3,48), mi(3,48), la(3,48), mi(3,48)]
seq_p+=perc_repeat

seq+=[('la', la(5,24), 'la', la(5,12), 'la', ti(5,12), 'la', do(6,24), 'la', ti(5,24), 'la', la(5, 72)), BL(24)]
seq_g+=[la(4,96), BK(72), do(5,12), mi(5,12), la(5,24), mi(5,24), la(4,96), BK(72), do(5,12), mi(5,12), do(5,48), BK(46), mi(5,46), BK(44), la(5,44)]
seq_b+=[la(3,48), mi(3,48), la(3,96)]
seq_p+=perc_repeat

doc.sing(seq, ZhaoYingzi, track)
doc.playNoteSeq(seq_g, Guitar, track_g)
doc.playNoteSeq(seq_b, Bass, track_b)
doc.playNoteSeq(seq_o, Ocarina, track_o)
doc.playBeatSeq(seq_p, perc_list, track_p)
doc.playBeatSeq(seq_e, effect_list, track_effect)

doc.setTrackVolume(track_g, 0.5)
doc.setTrackVolume(track_b, 0.5)
doc.setTrackVolume(track_o, 0.5)
doc.setTrackVolume(track_p, 0.5)
doc.setTrackVolume(track_effect, 0.5)

doc.meteor()
doc.mixDown('TaLang.wav')
```





