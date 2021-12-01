# ScoreDraft

ScoreDraft 是一个音乐和歌唱合成系统，它提供了基于Python的乐谱创作（或录入）界面。

目前，它集成了下列几种合成引擎：

* 乐器合成
  - SimpleInstruments: 基于简单数学函数，如正弦波
  - KarplusStrong: 基于纯算法的吉他模拟器
  - BasicSamplers: 由单个或多个波形文件采样生成音符
  - SoundFont2
* 打击乐合成
  - BasicSamples
  - SoundFont2 (GM 鼓轨)
* 语音合成
  - VoiceSampler: 采用类PSOLA的算法采样语音样本，提供了名为UtauDraft的前端来兼容UTAU音源

系统框架是开放的，极易扩展。

PCMPlayer 提供了一个简单的音频播放和可视化工具。

Meteor 提供了更高级的可视化功能，可以可视化用来生成音乐的音符和音节输入。

下面这个例子展示了通过ScoreDraft来合成一小段音乐可以多么简单。

```Python
    import ScoreDraft
    from ScoreDraft.Notes import *

    doc=ScoreDraft.Document()

    seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

    doc.playNoteSeq(seq, ScoreDraft.Piano())
    doc.mixDown('twinkle.wav')
```

从版本 1.0.3 开始，ScoreDraft 支持一种基于 YAML 的输入格式，例如：

```yaml
# test.yaml
score:
    staffs:
        -
            relative: c''
            instrument: Piano()
            content: |
                c4 c g g a a g2
```

其中 content 部分使用 LilyPond 的语法。可以使用以下命令把 YAML 文件合成为wav:

```
# scoredraft -wav twinkle.wav test.yaml
```

更详细的使用说明和演示见: [https://fynv.github.io/ScoreDraft/](https://fynv.github.io/ScoreDraft/)

## 安装

ScoreDraft 现在可以由 PyPi 安装，支持64位的 Windows 和 Linux 系统。

```
# pip install scoredraft
```

已知问题：Linux方面只在Ubuntu20.04测试通过，已知在Ubuntu18.04上有问题。

## 编译

编译期依赖：

* CMake 3.0+

* Python3

* CUDA(可选): 如果没有CUDA可以去掉 CMake 的 "USE_CUDA" 选项

* FreeType: 
  
  - 已包含Windows开发库
  - Ubuntu: sudo apt install libfreetype-dev

* GLFW: 
  
  - 已包含源代码
  - Ubuntu: sudo apt install libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev

* PortAudio:
  
  - 已包含源代码
  - Ubuntu: sudo apt-get install libasound-dev libjack-dev

编译过程：

```
# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=../Test
# make
# make install
```

运行期依赖：

* Python3 
* cffi
* 播放器依赖于X.org和ALSA驱动
* MusicXML 和 LilyPond 支持依赖于 xsdata, python_ly
* YAML 支持依赖于 pyyaml

## 采样和语音音源

为了避免麻烦，ScoreDraft 目前只包含最低限度的乐器和打击乐采样用来测试。PyPi安装的版本则不提供任何这些文件。

ScoreDraft 基于 Python 脚本的启动位置来搜索特定目录，以此建立乐器样本和语音音源的索引。

* InstrumentSamples 目录：wav 乐器样本
  
  - 可以包含子目录，每个子目录包含多个wav文件，共同来定义一个乐器的不同音高范围。
  - 对每个wav文件，可以有一个对应的freq文件来定义采样的音高

* PercussionSamples 目录：wav 打击乐样本

* SF2 目录：SoundFont2 乐器库文件

* UTAUVoice：UTAU 语音音源库，每个库一个子目录

用户需要自行下载和布署这些音源。以下推荐几个作者自己在用的音源。

* wav 乐器采样：
  
  - https://freewavesamples.com

* SoundFont2 乐器库
  
  - Arachno: http://www.arachnosoft.com/main/download.php?id=soundfont-sf2
  - SynthFontViena: http://www.synthfont.com/
  - 

* UTAU
  
  - uta(Japanese): default voice-bank that comes with UTAU
  - TetoEng(English): https://kasaneteto.jp/en/voicebank.html
  - Ayaka(Chinese): https://bowlroll.net/file/53297
  - Ayaka2 (Japanese): https://bowlroll.net/file/69898
  - WanEr(Chinese): http://lindayana.lofter.com/waner

## 版权协议

ScoreDraft 目前已正式基于MIT协议授权。

## 版本历史

ScoreDraft 实际上是我的第一个Python项目，由于一些设计问题，长期以来ScoreDraft无法用常规的Python流程来打包和发布。
直到2021年11月的这次重构，ScoreDraft才有了第一个正式版本。

在2018年曾经做过一次重构尝试，叫做[SingingGadgets](https://pypi.org/project/singinggadgets/)，只部分解决了问题。

在2021年11月的重构之后，ScoreDraft已经具有SingingGadgets和早期ScoreDraft的全部优点，因此SingingGadgets项目现已被关闭。

* 2021年12月01日. ScoreDraft 1.0.6 暴露更多底层函数
* 2021年11月29日. ScoreDraft 1.0.4, 1.0.5 更新YAML功能
* 2021年11月27日. ScoreDraft 1.0.3 增加了一个基于YAML的乐谱输入方案
* 2021年11月24日, ScoreDraft 1.0.2 加入对MusicXML和LilyPond的支持
* 2021年11月19日. ScoreDraft 1.0.0 & 1.0.1
* 2018年06月16日. SingingGadgets 0.0.3
