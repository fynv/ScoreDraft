ScoreDraft
================
ScoreDraft 是一个简洁的音乐和歌唱合成系统，它提供了基于Python的乐谱创作（或录入）界面。

从ScoreDraft的开发历史来看，这套代码的设计目标曾经经历一些调整。
最开始，这个软件是用来进行音乐实验的，不如研究纯律与平均律的和声差别。并且，最初的设计
曾经考虑用文本解析的方式录入乐谱，也就是，用户用特定的语法以文本方式来编写乐谱，ScoreDraft
通过C++来解析文本并合成音乐。后来，经过尝试用Python作为ScoreDraft的前端，发现对于
乐谱的录入来说有很大的好处，于是之后的开发均转向Python界面。

再后来，作者逐渐对于类似于将Vocaloid和Utau的歌唱功能加入这个系统产生兴趣。通过脚本的方式来
生成歌声似乎也是不错的体验。于是这方面的工作就成了近期的开发重点。

在歌唱功能的开发过程中，目前经历了两个阶段。第一个阶段，首先实现了一个简单的，只支持单独音的
歌唱引擎。这个引擎逐个音符地生成声音，不考虑音和音之间的过渡。这个引擎起名为"KeLa"，取自著名
的“金坷垃”。在第二个阶段中，尝试实现一个与各种UTAU音源（包括VCV和CVVC)尽可能兼容的引擎，名
为“UtauDraft”。

我也不知道这个项目之后会走向何方。总体来说这里的工具集将会不断的扩充，使大家可以以脚本的方式
来合成各种不同的波形。

下面这个例子展示了通过ScoreDraft来合成一小段音乐可以多么简单。

```Python

	import ScoreDraft
	from ScoreDraftNotes import *
	
	doc=ScoreDraft.Document()
	
	seq=[do(),do(),so(),so(),la(),la(),so(5,96)]
	
	doc.playNoteSeq(seq, ScoreDraft.Piano())
	doc.mixDown('twinkle.wav')

```

# 关于拷贝

本账号在GitHub上发布的全部源代码您都可以自由地使用和拷贝。这里的“自由”等同于BSD或MIT协议所
给予的许可。

但请注意，本项目在二进制部分包含混合内容。

* Qtxxx.dll 版权属于 [The Qt Company](https://www.qt.io/), 使用和拷贝需遵守LGPL协议。
* Alesis-Fusion-Bright-Acoustic-Piano-xx.wav, BassDrum.wav, ClosedHitHat.wav, Snare.wav来自
[https://freewavesamples.com](https://freewavesamples.com)
在使用和拷贝时请遵守其原始版权声明。
* 来自葛平音源和金坷垃jklex音源的少量内容，在使用和拷贝时请参照其原始版权声明
[http://utau.vocalover.com/](http://utau.vocalover.com/)

请勿使用ScoreDraft从事任何有违法律和道德的活动，一旦发生该情况， ScoreDraft的作者不负任何责任。
ScoreDraft作者也不对ScoreDraft的功能性和安全性做任何承诺。

# 内容结构

首先，您最需要的可能是: 

	/python_test 

这个目录.

此目录包含了 ScoreDraft Python层的全部内容。此外，由C++代码编译得到的核心部分也预先部署在此
目录下。预编译的二进制文件支持Windows x64 和Linux AMD64操作系统。因此，使用这些操作系统的
Python用户可以利用此目录进行快速的测试和二次开发而无需由代码编译。

随着开发进行，python_test目录下的内容将会被动态更新。这种情况下，作者认为本项目没有必要做正式
的打包发布，用户可以随时从python_test目录下自取所需。这里包含的音源样本将控制在极小的规模，
用户根据自己的需求可以自行获取和部署音源。

主要Python接口:

	/python_test/PyScoreDraft.pyd (or PyScoreDraft.so): 由C++部分导出的原始接口
	/python_test/ScoreDraft.py: ScoreDraft核心Python接口， 是对PyScoreDraft的封装
	/python_test/ScoreDraftNotes.py: 音符定义
	/python_test/ScoreDraftRapChinese.py: 用于中文四声的Rap辅助函数
	/python_test/print_generated_code.py: 打印由C++部分动态生成的Python代码，包含扩展接口
	/python_test/print_generated_code_summary.py: 打印动态Python代码的摘要

子目录:

	/python_test/Extensions: 对 PyScoreDraft 的扩展，C++模块
	/python_test/InstrumentSamples: 乐器音频样本，用于 InstrumentSampler 扩展
	/python_testPercussionSamples: 打击乐音频样本，用语 PercussionSampler 扩展
	/python_test/KeLaSamples: 用语 KeLa 歌唱引擎的语音样本
	/python_test/UTAUVoice: 符合UTAU标准的音源库，用于UtauDraft歌唱引擎，可以从Utau的voice目录下拷贝过来
	/python_test/Examples: 例子 

用于UtauDraft歌唱引擎的歌词转换器:

	/python_test/CVVCChineseConverter.py: 将汉语单独音歌词转为CVVCChinese歌词
	/python_test/TsuroVCVConverter.py: 将汉语单独音歌词转为樗式VCV歌词
	/python_test/JPVCVConverter.py: 将日语单独音歌词转为日语连续音歌词
	/python_test/TTEnglishConverter.py: 英文歌词（Delta式）转换器，仍需要TTEnglishInputHelper，但可以使用“for presamp”形式
	/python_test/TTLyricSet.data: TTEnglishConverter的数据文件, 由oto.ini转换得到
	/python_test/XiaYYConverter.py: 将汉语单独音歌词转为夏语遥式歌词

用Python写的各种测试样例:

	/python_test/Examples/Hello.py: 最简单的例子
	/python_test/Examples/piano_test.py: InstrumentSampler 测试
	/python_test/Examples/percussion_test.py: PercussionSampler 测试
	/python_test/Examples/FlyMeToTheMoon.py: fly me to the moon 纯律版
	/python_test/Examples/FlyMeToTheMoon_eq.py: fly me to the moon 平均律版
	/python_test/Examples/NaushikaRequiem.py: 那乌西卡安魂曲
	/python_test/Examples/KeLaTest.py: 一个使用金坷垃音源的小测试
	/python_test/Examples/GePi.py: 葛平圆周率10分钟版
	/python_test/Examples/GePi_unlimited.py: 葛平圆周率无限版
	/python_test/Examples/RapTest.py: Rap测试，朗读唐诗300首
	/python_test/Examples/tang300.py: 唐诗300首，RapTest的数据库
	/python_test/Examples/CVVCTest.py: 英文CVVC音源测试，测试所用音源:
		[kasaneteto, English Voicebank (CVVC) Voicebank](http://kasaneteto.jp/en/voicebank.html)
	/python_test/Examples/CVVCTest2.py: CVVCChinese 音源测试，测试所用音源:
		[三色あやかCVVChinese V1.0](https://bowlroll.net/file/53297)
	/python_test/Examples/VCVTest.py: 樗式中文VCV音源测试，测试所用音源：
		[綰儿] (http://lindayana.lofter.com/waner)
	/python_test/Examples/uta_test.py: 日语音源测试，测试所用音源:
		a. UTAU软件自带的"uta"音源
		b. [三色あやか 連続音V2.0](https://bowlroll.net/file/69898)
	/python_test/Examples/XiaYYTest.py: 夏语遥式中文CVVC测试 [夏语遥](http://www.voicemith.com/download.html)

python_test 之外的内容基本上是C++代码，您可以自行编译。

# 基于用例的使用说明

[新增 Jupyter notebook 幻灯片](https://github.com/fynv/ScoreDraft/blob/master/python_test/Examples/Singing%20through%20Python_zh.ipynb)

首先，为了使用ScoreDraft，所有的代码都需要

```Python

	import ScoreDraft

```
Python 空间的主要接口都在python_test/ScoreDraft.py中定义和注释。这里只基于例子给出简要的使用
说明。另外，只是使用的话，PyScoreDraft的接口就不用管了，ScoreDraft.py已经做了很好的封装。 


## 初始化 乐器(Intruments)/打击乐（Percussions）/歌手(Singers)

您可以运行 print_generated_code_summary.py 来查看有哪些可以使用的乐器/打击乐/歌手类：

```

	Instruments:
	0: PureSin
	1: Square
	2: Triangle
	3: Sawtooth
	4: NaivePiano
	5: BottleBlow
	6: Ah
	7: Cello
	8: Lah
	9: String
	10: Violin
	11: Piano
	
	Percussions:
	0: TestPerc
	1: BassDrum
	2: Snare
	
	Singers:
	0: GePing
	1: KeLaTest
	2: Up
	
	Interfaces:
	0: WriteNoteSequencesToMidi
	1: PlayTrackBuffer

```

上面列出的每一项都对应一个动态生成的初始化函数，例如，您可以通过下面的代码来初始化一个大提琴
乐器：

```Python

	Cello1= ScoreDraft.Cello()

```

如果您运行 print_generated_code.py 您将可以看到全部的动态代码和相关的注释。


## 用乐器来演奏

添加代码:

```Python

	from ScoreDraftNotes import *

```

这样您就可以得到 do() re() mi() fa() so().. 的定义了。

ScoreDraft 核心接口本身接受的是“频率”参数，上面这些函数则帮您把音乐意义上的“音高”转换为物理意义上的
“频率”， 也就是相当于律制系统的定义。ScoreDraftNotes中默认采用的是十二平均律，您可以通过修改
频率来定义自己的律制系统。

对于乐器演奏，您可以按如下格式定义一个音符序列：

```Python

	seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

```
每个“音符”都有两个参数，八度和时值。默认的八度值是5，代表中央八度。（这个和MIDI的标准一致，和UTAU不一致，UTAU的中央八度是4）。默认是时值是48， 代表1拍。

ScoreDraftNotes 还定义了两个特殊音符 BL() 和 BK()。 BL()代表一个空拍，将光标前移，BK()代表一个退格，将光标后移，这样后面写的音符就有可能和前面写的音符出现发生重合。例如一个大三和弦可以写作：

```Python

	seq=[do(5,48), BK(48), mi(5,48), BK(48), so(5,48)]

``` 

然后，初始化一个“文档”对象:

```Python

	doc=ScoreDraft.Document()

```

使用某个乐器，您可以把前面定义的音符序列“演奏”到文档当中：

```Python

	doc.playNoteSeq(seq, ScoreDraft.Piano())

```

调用doc.mixDown, 混音得到 .wav file:

```Python

	doc.mixDown('twinkle.wav')

```

### 乐器采样器

乐器采样器（InstrumentSampler）扩展允许用户通过一个或多个乐器音频样本加入 "InstrumentSamples"
目录来扩展乐器集合。

乐器采样器支持单采样和多采样两种模式。

对于单采样，只需将单个.wav文件加入 "InstrumentSamples"目录，此时去掉扩展名的文件名将被系统
注册为该乐器的类名。

对于多采样，首先在"InstrumentSamples"目录中添加一个子目录，该子目录的名字将被系统
注册为该乐器的类名。然后，将多个.wav文件放入新加的子目录中。这些音频样本应覆盖一定的音高范围。
采样器将通过在这些音高之间基于目标音高来插值得到最终的合成结果。

这里的.wav文件必须是1个或2个通道的16bit PCM格式

乐器采样器的算法是通过对样本的简单拉伸来改变音高的。因此使用的音频样本应具有足够的长度。


## 打击乐演奏

对于打击乐演奏，首先您应该考虑选择哪些打击乐器来组成一个打击乐组。
例如，我选择低音鼓和小军鼓：

```Python

	BassDrum=ScoreDraft.BassDrum()
	Snare=ScoreDraft.Snare()
	
	perc_list= [BassDrum, Snare]

```

然后，为了方便编写乐谱，定义两个辅助函数：


```Python

	def dong(duration=48):
		return (0,duration)
	
	def ca(duration=48):
		return (1,duration)

```
上面的声明意味着以后“dong()”代表着0号打击乐器（BassDrum）的声音，"ca()"，代表着1号打击乐器(Snare)的声音。那么，一个节奏序列就可以如下定义了：


```Python

	seq = [dong(), ca(24), dong(24), dong(), ca(), dong(), ca(24), dong(24), dong(), ca()]

```

和乐器演奏同样地，下面初始化一个“文档”对象，用打击乐组将节奏序列演奏到文档，然后混音：

```Python

	doc=ScoreDraft.Document()
	doc.setTempo(120)
	doc.playBeatSeq(seq, perc_list)
	doc.mixDown('test_perc.wav')

```

### 打击乐采样器

打击乐采样器（PercussionSampler）扩展允许用户通过将单个的音频样本添加到 "PercussionSamples" 目录来扩展打击乐集合。去掉扩展名的文件名将被系统注册为该打击乐器的类名。

这里的.wav文件必须是1个或2个通道的16bit PCM格式。

打击乐采样器在使用样本时不做任何修改，直接添加包络。因此使用的音频样本应具有足够的长度。


## 唱歌

ScoreDraft 提供的唱歌界面与乐器/打击乐演奏界面比较类似。

一个歌唱序列相比乐器演奏中的音符序列较为复杂，例如：


```Python

	seq= [ ("jin_f1", do(5,24), ti(4,24), do(5,24)), ("ke_f1", re(5,24), "la_f1", mi(5,24)) ]

```

每个“歌唱片段”（由逗号分隔的第一级元素）包含一个或多个作为歌词的字符串。每个歌词后可以跟１个或多个音符。这些音符均对应于前面的歌词，以此来实现音节内的多音高。每个“歌唱片段”内的所有音节（歌词＋音符）都应连续演唱，除非遇到空拍 BL() 或倒退 BK()，那时系统将不得不把把一个歌唱片段分解成多个歌唱片段来处理。

下面初始化一个“文档”对象，选择一个歌手来演唱序列到文档，然后混音：

```Python

	doc=ScoreDraft.Document()
	doc.sing(seq, ScoreDraft.KeLaTest())
	doc.mixDown('KeLa.wav')

```

用户也可以将用于乐器的普通音符和歌唱片段混合。在该情况下，这些音符将使用歌手的默认歌词来演唱。

反过来也是可以的，你可以把一个歌唱序列交给一个乐器来演奏。这种情况下，歌词信息将被忽略，只有音符被演奏。

目前歌唱系统还支持Rap，歌唱序列可以包含Rap片段，像下面序列中第二个片段：


```Python

	seq= [ ("jin_f1", do(5,48)), ("ke_f1", 48, 1.0, 1.0, "la_f1", 48, 1.0, 0.5) ]

```

这里的Rap片段 ("ke_f1", 48, 1.0, 1.0, "la_f1", 48, 1.0, 0.5) 中, 每个歌词后面跟着三个数字。
第一个数字是该音节的时长，和音符一样用整数表示。后面两个浮点数代表一个音节的起始频率和结束频率。

在ScoreDraftRapChinese.py中提供了一个工具函数，可以用来帮助生成中文四声的Rap。有了它的帮助，
上面这段Rap可以如下这样写：


```Python

	from ScoreDraftRapChinese import *

	seq= [ ("jin_f1", do(5,48)), RapTone("ke_f1", 1, 48)+RapTone("la_f1", 4, 48)) ]

```


### KeLa 引擎

KeLa 引擎是一个比较简单的歌唱引擎，作为一个扩展提供。它通过（语音学地）拉伸单个的语音样本来逐个
合成音符。支持类似UTAU中单独音的功能。

用于 KeLa 引擎的语音样本应放在 "KeLaSamples"目录（的各个子目录）下，每个子目录用来定义一个歌手类。
每个子目录的名字将被系统注册为该歌手的类名， 如上面的 "KeLaTest"。每个子目录包含多个.wav文件，
不包含扩展名的文件名将被作为歌词在歌唱片段中使用。

与乐器采样器和打击乐采样器不同的是，KeLa引擎可以使用短小的语音样本。因为算法会提取样本的特征来生成
任意长度的音符，样本本身不需要很长。录音的时候只需要注意音调尽量平平稳。

这里的.wav文件同样必须是1个或2个通道的16bit PCM格式。
另外，由于目前没有考虑不同采样频率间的转换问题，如果使用了非44100的.wav文件，结果可能不正确。


### UtauDraft 引擎

UtauDraft 引擎试图尽可能支持UTAU的各种音源，包括单独音，连续音，VCV, CVVC等。引擎会读取音源
的oto.ini和.frq来提取音源的特征。如果有prefix.map，引擎也会读取这个文件来进行样本音高的选择。

UTAU 的音源可以直接放在"UTAUVoice"目录下。"UTAUVoice"的每个子目录定义一个歌手类。
歌手类的类名为子目录名加上"_UTAU"后缀。例如，子目录名为"GePing"，则对应歌手类的类名为"GePing_UTAU"。如果音源目录原来的名字不适合用作Python的变量名，那么用户应对目录名进行必要的
修改以避免发生Python解析错误。

音源中的.wav文件同样必须是1个或2个通道的16bit PCM格式。
另外，由于目前没有考虑不同采样频率间的转换问题，如果使用了非44100的.wav文件，结果可能不正确。

当使用 UtauDraft 引擎时，歌唱片段中使用的歌词应以oto.ini中的定义为准，正如在UTAU中那样。

当使用VCV或CVVC等类型的音源时，为了简化歌词录入，可以由用户在Python空间实现一个配合该音源的
拆词函数，然后通过ScoreDraft接口 "ScoreDraft.UtauDraftSetLyricConverter(singer, LyricConverterFunc)"将该函数设置给歌手使用。例如：

```Python

	import ScoreDraft
	import CVVCChineseConverter

	Ayaka = ScoreDraft.Ayaka_UTAU()
	ScoreDraft.UtauDraftSetLyricConverter(Ayaka, CVVCChineseConverter.CVVCChineseConverter)

```

这里，拆词函数的实现在CVVCChineseConverter.py中，我们将它设置给了Ayaka这个歌手。设置之后，在
歌唱片段中就可以使用单独音的歌词了。

拆词函数应具有以下的形式：

```Python

	def LyricConverterFunc(LyricForEachSyllable):
		...
		return [(lyric1ForSyllable1, weight11, isVowel11, lyric2ForSyllable1, weight21, isVowel21...  ),(lyric1ForSyllable2, weight12, isVowel12, lyric2ForSyllable2, weight22, isVowel22...), ...]

```

输入参数'LyricForEachSyllable' 是歌唱片段中输入的歌词列表 [lyric1, lyric2, ...], 每个歌词
对应一个音节。拆词函数将每个输入歌词转换为1个或多个歌词，来瓜分原歌词的时值。输出的时候，要给每个
分解后的歌词设置一个权重，以指示分解后的歌词在原歌词的时值中所占的比例。另外还需要提供一个bool值isVowel只是分离出来的这个部分是否包含原音节的元音部分。

目前已经实现了若干拆词函数的实例，可以直接使用它们，也可以在实现其他拆词函数是作为参考。详见：
CVVCChineseConverter.py, TsuroVCVConverter.py 和 JPVCVConverter.py.

# 关于编译

如果要自己编译ScoreDraft, 用户需要首先安装:

* CMake 3.0+
* Python3
* Qt5 (这个只是"QtPCMPlayer"需要，如果您不需要，您可以把它从/CMakeLists.txt中移除，这样就可以不安装Qt5了）

运行 CMake 来为您的编译器生成makefiles或project files，然后即可编译。

在运行CMake时，建议：

* 使用 /build 作为编译目录
* 将 CMAKE_INSTALL_PREFIX 设置为 /python_test 所在位置
