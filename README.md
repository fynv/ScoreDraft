[-> 中文Readme](https://github.com/fynv/ScoreDraft/blob/master/README_cn.md)

ScoreDraft
================
ScoreDraft is a simple music/singing synthesizer that provides a Python based 
score authoring interface. 

Historically, there has been some changes in the design goal of the software.
At the beginning, the software was intendend to be used for music experiments 
like just intonation, and in the first design, I was intending to use text 
files as the means of authoring, i.e., the C++ program reads a text file, 
parse the notes and generates an audio file. However, after trying using
Python as a script layer, it has shown very much benefit comparing to text 
parsing. So I moved quickly to Python.

Then, I became interested in extending ScoreDraft for singing synthesis, simliar 
to Vocaloid and UTAU, but using script, and this has become the focus of the 
developent of ScoreDraft recently. In the first stage, I implemented a simple
voice synthesizing engine that generates voice note by note from individual voice
samples. This engine is called "KeLa", named after the famous video on Bilibili.
In the second stage, I try to build a new engine that can make use of voice-banks 
made for UTAU, including those CVVC and VCV voice-banks. The new engine is called 
"UtauDraft". The name only indicates the intension to be compatible with UTAU voice
banks.

I'm really not sure where ScoreDraft will go in the furture. Basically, here, 
I'm building a big set of utilities to allow different kinds all waveforms to
be generated and mixed together through scripts. 

The following example shows how easily a piece of musical sound can be generated
using ScoreDraft.


```Python

	import ScoreDraft
	from ScoreDraftNotes import *
	
	doc=ScoreDraft.Document()
	
	seq=[do(),do(),so(),so(),la(),la(),so(5,96)]
	
	doc.playNoteSeq(seq, ScoreDraft.Piano())
	doc.mixDown('twinkle.wav')

```

# About Copying

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


# Repository Contents

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

Main Python interfaces:

	/python_test/PyScoreDraft.pyd (or PyScoreDraft.so): C++ exported original interfaces
	/python_test/ScoreDraft.py: core interfaces, a wrapper layer for PyScoreDraft.pyd
	/python_test/ScoreDraftNotes.py: note definitions
	/python_test/ScoreDraftRapChinese.py: define utilities to generate Mandarin Chinese 4 tone rap.
	/python_test/print_generated_code.py: list Python code dynamically generated from C++ 
	/python_test/print_generated_code_summary.py: list summary of the generated code 

Sub-directories:

	/python_test/Extensions: binary extensions to PyScoreDraft
	/python_test/InstrumentSamples: audio samples for InstrumentSampler extension
	/python_test/PercussionSamples: audio samples for PercussionSampler extension
	/python_test/KeLaSamples: audio samples for KeLa voice engine 
	/python_test/UTAUVoice: deploy your UTAU voice-banks here, the UtauDraft voice engine will use them
	/python_test/Examples: examples 

Lyric converters used for UtauDraft voice engine:

	/python_test/CVVCChineseConverter.py: converting PinYin lyrics for each syllable into CVVCChinese lyrics
	/python_test/TsuroVCVConverter.py: converting PinYin lyrics for each syllable into Tsuro style Chinese VCV lyrics
	/python_test/JPVCVConverter.py: converting Japanese 単独音 lyrics to 連続音 lyrics
	/python_test/TTEnglishConverter.py: simplifing the input of english lyrics for Kasane Teto English Voicebank. You'll still need TTEnglishInputHelper, but can use the "for presamp" form of phonetics.
	/python_test/TTLyricSet.data: data file for TTEnglishConverter, dumped from the oto.ini
	of the voice-bank.
	/python_test/XiaYYConverter.py: converting PinYin lyrics for each syllable into lyrics specifically for XiaYuYao voice-bank

Examples:

	/python_test/Examples/Hello.py: the simplest example
	/python_test/Examples/piano_test.py: test of instrument sampler
	/python_test/Examples/percussion_test.py: test of percussion sampler
	/python_test/Examples/FlyMeToTheMoon.py: fly me to the moon in just-intonation
	/python_test/Examples/FlyMeToTheMoon_eq.py: fly me to the moon in equal-temperment
	/python_test/Examples/NaushikaRequiem.py: Naushika Requiem using a simple voice, with accompany
	/python_test/Examples/KeLaTest.py: a test using the Jin-Ke-La voice bank
	/python_test/Examples/GePi.py: a 10 minute rendering of the Pi song using GePing voice-bank
	/python_test/Examples/GePi_unlimited.py: unlimited real-time generation/playback of the Pi song. 
	/python_test/Examples/RapTest.py: test of the the Rap function by singing Chinese Peoms. Need the full-set of "GePing" voice-bank
	/python_test/Examples/tang300.py: the database for RapTest.py
	/python_test/Examples/CVVCTest.py: a test using a English CVVC voice-bank, specifically:
		[kasaneteto, English Voicebank (CVVC) Voicebank](http://kasaneteto.jp/en/voicebank.html)
	/python_test/Examples/CVVCTest2.py: a test using a CVVCChinese voice-bank, specifically:
		[三色あやかCVVChinese V1.0](https://bowlroll.net/file/53297)
	/python_test/Examples/VCVTest.py: a test using a Tsuro style Chinese VCV voice-bank, specifically:
		[綰儿] (http://lindayana.lofter.com/waner)
	/python_test/Examples/uta_test.py: a test using Japanese voice-banks, specifically:
		a. the "uta" voice-bank that comes with UTAU
		b. [三色あやか 連続音V2.0](https://bowlroll.net/file/69898)
	/python_test/Examples/XiaYYTest.py: a test using XiaYuYao voice-bank, which is a 
        special form of CVVCChinese. [XiaYuYao](http://www.voicemith.com/download.html)

Contents outside of python_test are basically C++ source-code files managed with CMake, 
which are used to build the binaries.

# Usage Introduction

Usage introduction of ScoreDraft has been moved to:

[http://scoredraft.org/index.php/introduction-to-scoredraft/](http://scoredraft.org/index.php/introduction-to-scoredraft/)

Chinese version (中文使用介绍）:

[http://scoredraft.org/index.php/scoredraft-introduction-zh/](http://scoredraft.org/index.php/scoredraft-introduction-zh/)

# About building

To build ScoreDraft from source-code, you need to install:

* CMake 3.0+
* Python3
* Qt5 (Needed by "QtPCMPlayer“ and "Meteor" extensions. You can remove them from /CMakeLists.txt if you don't need them. Then you don't need Qt5)

Run CMake to generate makefiles/project files for your system and build.
You are recommanded to:

* use the /build directory as you building directory
* use /python_test as your CMAKE_INSTALL_PREFIX

