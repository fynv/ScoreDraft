[-> 中文Readme](https://github.com/fynv/ScoreDraft/blob/master/README_cn.md)

ScoreDraft
================
ScoreDraft is a simple music/singing synthesizer that provides a Python based 
score authoring interface. 

Historically, there have been changes in the design goal of ScoreDraft.
The software was originally intendend to be used for music experiments 
such as just intonation. For the initial design, I was intending to use text 
files as the means of authoring, i.e., the C++ program reads a text file, 
parse the notes and generates an audio file. However, after using
Python as a script layer, it has shown very much benefit comparing to text 
parsing. So I moved quickly to Python.

I became interested in extending ScoreDraft for singing synthesis, simliar 
to Vocaloid and UTAU, but using script. Since then singing synthesis 
has become the focus of development. In the first stage, I implemented a simple
voice synthesizing engine that generates a voice note by note from individual voice
samples. This engine is called "KeLa", named after the famous video on Bilibili.
In the second stage, I built a new engine makes use of voice-banks 
made for UTAU, including CVVC and VCV voice-banks. The new engine is called 
"UtauDraft". The name only indicates the intension to be compatible with UTAU voice
banks.

I'm not sure where ScoreDraft will go in the furture. From here I'm building 
a big set of utilities to allow different kinds all waveforms to
be generated and mixed together through scripts. 

The following example shows how easily a piece of musical sound can be generated
using ScoreDraft.


```Python

	import ScoreDraft
	from ScoreDraft.Notes import *
	
	doc=ScoreDraft.Document()
	
	seq=[do(),do(),so(),so(),la(),la(),so(5,96)]
	
	doc.playNoteSeq(seq, ScoreDraft.Piano())
	doc.mixDown('twinkle.wav')

```

## About Copying

I'm intending to make all my source-code contribution to GitHub totally free to 
use, like the permission granted by a BSD or MIT license.

However, the repository itself contains mixed content. I wrote all the 
C/C++/Python/CMake source code. Basically, you can use and copy these files freely.
There are dlls from Qt5 in the binary part, which are in LGPL license. 
Be careful when using them. There are also a few audio samples from:

[https://freewavesamples.com](https://freewavesamples.com)

[http://utau.vocalover.com/](http://utau.vocalover.com/)

Please refer to the original copyright information when using/copying these files.

DON'T USE THE CONTENT OF THIS REPOSITORY TO DO ANY BAD THINGS.
THE AUTHOR OF SCOREDRAFT DOES NOT HAVE ANY RESPONSIBILITY IN SUCH CASE.
NO WARRANTY OF ANY KIND FROM THE AUTHOR.


## Repository Contents

First please check: 

	/python_test 

directory.

The above directory contains the whole Python layer of ScoreDraft. Since the core
functions are implemented in C++, I also pre-deployed binaries built for Windows x64
and Linux Amd64 in that directory, so users of these systems can have a really quick 
evaluation without building ScoreDraft from source code.

These files are being dynamically updated as the development progresses. In that way
I think there will be no need to make any formal releases. Audio sample files will
be kept at small scale here. Users can extend that part themselves.

Sub-directories:

	/python_test/ScoreDraft: the ScoreDraft package 
	/python_test/ScoreDraft/Extensions: binary extensions
	/python_test/ScoreDraft/InstrumentSamples: audio samples for InstrumentSampler extension
	/python_test/ScoreDraft/PercussionSamples: audio samples for PercussionSampler extension
	/python_test/ScoreDraft/KeLaSamples: audio samples for KeLa voice engine 
	/python_test/ScoreDraft/UTAUVoice: deploy your UTAU voice-banks here, the UtauDraft voice engine will use
	/python_test/zh_TTS: a simple Chinese TTS demo using pypinyin + ScoreDraft   

Examples:

	/python_test/Hello.py: the simplest example
	/python_test/piano_test.py: test of instrument sampler
	/python_test/percussion_test.py: test of percussion sampler
	/python_test/FlyMeToTheMoon.py: fly me to the moon in just-intonation
	/python_test/FlyMeToTheMoon_eq.py: fly me to the moon in equal-temperment
	/python_test/NaushikaRequiem.py: Naushika Requiem using a simple voice, with accompany
	/python_test/KeLaTest.py: a test using the Jin-Ke-La voice bank
	/python_test/GePi.py: a 10 minute rendering of the Pi song using GePing voice-bank
	/python_test/GePi_unlimited.py: unlimited real-time generation/playback of the Pi song. 
	/python_test/RapTest.py: test of the the Rap function by singing Chinese Peoms. Need the full-set of "GePing" voice-bank
	/python_test/tang300.py: the database for RapTest.py
	/python_test/CVVCTest.py: a test using a English CVVC voice-bank, specifically:
		[kasaneteto, English Voicebank (CVVC) Voicebank](http://kasaneteto.jp/en/voicebank.html)
	/python_test/CVVCTest2.py: a test using a CVVCChinese voice-bank, specifically:
		[三色あやかCVVChinese V1.0](https://bowlroll.net/file/53297)
	/python_test/VCVTest.py: a test using a Tsuro style Chinese VCV voice-bank, specifically:
		[綰儿] (http://lindayana.lofter.com/waner)
	/python_test/uta_test.py: a test using Japanese voice-banks, specifically:
		a. the "uta" voice-bank that comes with UTAU
		b. [三色あやか 連続音V2.0](https://bowlroll.net/file/69898)
	/python_test/XiaYYTest.py: a test using XiaYuYao voice-bank, which is a 
        special form of CVVCChinese. [XiaYuYao](http://www.voicemith.com/download.html)

Contents outside of python_test are basically C++ source-code files managed with CMake, 
which are used to build the binaries.

## Usage Introduction

English introduction

[http://scoredraft.org/index.php/introduction-to-scoredraft/](http://scoredraft.org/index.php/introduction-to-scoredraft/)

Chinese introduction (中文使用介绍）:

[http://scoredraft.org/index.php/scoredraft-introduction-zh/](http://scoredraft.org/index.php/scoredraft-introduction-zh/)

## About building

To build ScoreDraft from source-code, you need to install:

* CMake 3.0+
* Python3
* Qt5 (Needed by "QtPCMPlayer“ and "Meteor" extensions. You can remove them from /CMakeLists.txt if you don't need them. Then you don't need Qt5)
* CUDA (Optionally needed by UtauDraft, can be disabled by setting "USE_CUDA" to false in /UtauDraft/CMakeLists.txt)

Run CMake to generate makefiles/project files for your system and build.
You are recommanded to:

* use the /build directory as you building directory
* use /python_test as your CMAKE_INSTALL_PREFIX

