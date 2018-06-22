ScoreDraft
================
ScoreDraft 是一个音乐和歌唱合成系统，它提供了基于Python的乐谱创作（或录入）界面。

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
	from ScoreDraft.Notes import *
	
	doc=ScoreDraft.Document()
	
	seq=[do(),do(),so(),so(),la(),la(),so(5,96)]
	
	doc.playNoteSeq(seq, ScoreDraft.Piano())
	doc.mixDown('twinkle.wav')

```

## 关于拷贝

本账号在GitHub上发布的全部源代码您都可以自由地使用和拷贝。这里的“自由”等同于BSD或MIT协议所
给予的许可。

ScoreDraft 目前已正式基于MIT协议授权。

注意事项：

* 涉及到SoundFont2支持的部分移植了来自TinySoundFont(https://github.com/schellingb/TinySoundFont) 的代码，在MIT协议框架下使用。
* Qtxxx.dll 版权属于 [The Qt Company](https://www.qt.io/), 使用和拷贝需遵守LGPL协议。
* Alesis-Fusion-Bright-Acoustic-Piano-xx.wav, BassDrum.wav, ClosedHitHat.wav, Snare.wav来自
[https://freewavesamples.com](https://freewavesamples.com)
在使用和拷贝时请遵守其原始版权声明。
* 来自葛平音源和金坷垃jklex音源的少量内容，在使用和拷贝时请参照其原始版权声明
[http://utau.vocalover.com/](http://utau.vocalover.com/)

请勿使用ScoreDraft从事任何有违法律和道德的活动，一旦发生该情况， ScoreDraft的作者不负任何责任。
ScoreDraft作者也不对ScoreDraft的功能性和安全性做任何承诺。

## 内容结构

首先，您最需要的可能是: 

	/python_test 

这个目录.

此目录包含了 ScoreDraft Python层的全部内容。此外，由C++代码编译得到的核心部分也预先部署在此
目录下。预编译的二进制文件支持Windows x64 和Linux AMD64操作系统。因此，使用这些操作系统的
Python用户可以利用此目录进行快速的测试和二次开发而无需由代码编译。

随着开发进行，python_test目录下的内容将会被动态更新。这种情况下，作者认为本项目没有必要做正式
的打包发布，用户可以随时从python_test目录下自取所需。这里包含的音源样本将控制在极小的规模，
用户根据自己的需求可以自行获取和部署音源。


子目录:

	/python_test/ScoreDraft: ScoreDraft 包 
	/python_test/ScoreDraft/Extensions: C++扩展模块
	/python_test/ScoreDraft/InstrumentSamples: 乐器音频样本，用于 InstrumentSampler 扩展
	/python_test/ScoreDraft/PercussionSamples: 打击乐音频样本，用语 PercussionSampler 扩展
	/python_test/ScoreDraft/KeLaSamples: 用语 KeLa 歌唱引擎的语音样本
	/python_test/ScoreDraft/UTAUVoice: 符合UTAU标准的音源库，用于UtauDraft歌唱引擎，可以从Utau的voice目录下拷贝过来
	/python_test/zh_TTS: 一个简单的中文TTS，基于pypinyin + ScoreDraft

测试样例:

	/python_test/Hello.py: 最简单的例子
	/python_test/piano_test.py: InstrumentSampler 测试
	/python_test/percussion_test.py: PercussionSampler 测试
	/python_test/FlyMeToTheMoon.py: fly me to the moon 纯律版
	/python_test/FlyMeToTheMoon_eq.py: fly me to the moon 平均律版
	/python_test/NaushikaRequiem.py: 那乌西卡安魂曲
	/python_test/KeLaTest.py: 一个使用金坷垃音源的小测试
	/python_test/GePi.py: 葛平圆周率10分钟版
	/python_test/GePi_unlimited.py: 葛平圆周率无限版
	/python_test/RapTest.py: Rap测试，朗读唐诗300首
	/python_test/tang300.py: 唐诗300首，RapTest的数据库
	/python_test/CVVCTest.py: 英文CVVC音源测试，测试所用音源:
		[kasaneteto, English Voicebank (CVVC) Voicebank](http://kasaneteto.jp/en/voicebank.html)
	/python_test/CVVCTest2.py: CVVCChinese 音源测试，测试所用音源:
		[三色あやかCVVChinese V1.0](https://bowlroll.net/file/53297)
	/python_test/VCVTest.py: 樗式中文VCV音源测试，测试所用音源：
		[綰儿] (http://lindayana.lofter.com/waner)
	/python_test/uta_test.py: 日语音源测试，测试所用音源:
		a. UTAU软件自带的"uta"音源
		b. [三色あやか 連続音V2.0](https://bowlroll.net/file/69898)
	/python_test/XiaYYTest.py: 夏语遥式中文CVVC测试 [夏语遥](http://www.voicemith.com/download.html)

python_test 之外的内容基本上是C++代码，您可以自行编译。

## 使用说明

英文使用说明：

[http://scoredraft.org/index.php/introduction-to-scoredraft/](http://scoredraft.org/index.php/introduction-to-scoredraft/)


中文使用说明：

[http://scoredraft.org/index.php/scoredraft-introduction-zh/](http://scoredraft.org/index.php/scoredraft-introduction-zh/)

## 关于编译

如果要自己编译ScoreDraft, 用户需要首先安装:

* CMake 3.0+
* Python3
* Qt5 (这个只在可视化扩展"QtPCMPlayer"和"Meteor"中需要，如果您不需要这些扩展，您可以把它们从/CMakeLists.txt中移除，这样就可以不安装Qt5了）
* CUDA (UtauDraft默认启用CUDA加速，可以在/UtauDraft/CMakeLists.txt中将USE_CUDA设置为false来禁用)

运行 CMake 来为您的编译器生成makefiles或project files，然后即可编译。

在运行CMake时，建议：

* 使用 /build 作为编译目录
* 将 CMAKE_INSTALL_PREFIX 设置为 /python_test 所在位置
