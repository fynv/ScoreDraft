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

Then, I became interested in extending ScoreDraft for singing sythesis, simliar 
to Vocaloid and UTAU, but using script, and this has become the focus of the 
developent of ScoreDraft recently. In the first stage, I implemented a simple
voice synthesizing engine that generates voice note by note from individual voice
samples. This engine is called "KeLa", named after the famous video on Bilibili.
In the second stage, I try to build a new engine that can make use of voice-banks 
made for UTAU, including those CVVC and VCV voice-banks. The new engine is called "UtauDraft". The name only indicates the intension to be compatible with UTAU voice
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
	/python_test/ScoreDraftArranging.py: "arranging" interface, will be useful in the future
	/python_test/print_generated_code.py: list Python code dynamically generated from C++ 
	/python_test/print_generated_code_summary.py: list summary of the generated code 

Sub-directories:

	/python_test/Extensions: binary extensions to PyScoreDraft
	/python_test/InstrumentSamples: audio samples for InstrumentSampler extension
	/python_testPercussionSamples: audio samples for PercussionSampler extension
	/python_test/KeLaSamples: audio samples for KeLa voice engine 
	/python_test/UTAUVoice: deploy your UTAU voice-banks here, the UtauDraft voice engine will use them 

Lyric converters used for UtauDraft voice engine:

	/python_test/CVVCChineseConverter.py: converting PinYin lyrics for each syllable into CVVCChinese lyrics
	/python_test/TsuroVCVConverter.py: converting PinYin lyrics for each syllable into Tsuro style Chinese VCV lyrics
	/python_test/JPVCVConverter.py: converting Japanese 単独音 lyrics to 連続音 lyrics

Tests written in Python:

	/python_test/Hello.py: the simplest test
	/python_test/piano_test.py: test of instrument sampler
	/python_test/percussion_test.py: test of percussion sampler
	/python_test/FlyMeToTheMoon.py: fly me to the moon in just-intonation
	/python_test/FlyMeToTheMoon_eq.py: fly me to the moon in equal-temperment
	/python_test/NaushikaRequiem.py: Naushika Requiem using a simple voice, with accompany
	/python_test/KeLaTest.py: a test using the Jin-Ke-La voice bank
	/python_test/GePi.py: a 10 minute rendering of the Pi song using GePing voice-bank
	/python_test/GePi_unlimited.py: unlimited real-time generation/playback of the Pi song.
	/python_test/RapTest.py: test of the the Rap function by singing Chinese Peoms
	/python_test/tang300.py: the database for RapTest.py
	/python_test/CVVCTest.py: a test using a English CVVC voice-bank, specifically:
		[kasaneteto, English Voicebank (CVVC) Voicebank](http://kasaneteto.jp/en/voicebank.html)
	/python_test/CVVCTest2.py: a test using a CVVCChinese voice-bank, specifically:
		[三色あやかCVVChinese V1.0](https://bowlroll.net/file/53297)
	/python_test/uta_test.py: a test using Japanese voice-banks, specifically:
		a. the "uta" voice-bank that comes with UTAU
		b. [三色あやか 連続音V2.0](https://bowlroll.net/file/69898)

Contents outside of python_test are basically C++ source-code files managed with CMake, 
which are used to build the binaries.

# Usage by Examples

First

```Python

	import ScoreDraft

```

Python space interfaces are defined and documented in python_test/ScoreDraft.py.
Here I only give a brief introduction by examples. And, forget about the PyScoreDraft 
interface. All of the interfaces are properly wrapped by ScoreDraft.py. You won't
need it unless you are developing an extension of PyScoreDraft in C++.

## Initialization of Intruments/Percussions/Singers

Run print_generated_code_summary.py to get a list of all availabe instrument/percussion/singer classes:

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

Each of them is corresponding to a "generated" initializer function. For example you can initialize a 
Cello instrument by:

```Python

	Cello1= ScoreDraft.Cello()

```

Run print_generated_code.py and you will see a full list of the generated code with document.

## Instrument Play 

Add:

```Python

	from ScoreDraftNotes import *

```

So you can get a list of definations of do() re() mi() fa() so()..
(The default frequencies are tuned for 12-equal temperament, but you can change it)

For instrument play, write your note sequence like:

```Python

	seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

```

Each note has 2 parameters, octave and duration. The default octave is 5, which means the center octave.
The default value of duration is 48, which means 1 beat.

Then create a document object:

```Python

	doc=ScoreDraft.Document()

```

Play the notes to the document using some instrument:

```Python

	doc.playNoteSeq(seq, ScoreDraft.Piano())

```

Mix-down the generated tracks to a .wav file:

```Python

	doc.mixDown('twinkle.wav')

```

### Instrument Sampler

The instrument sampler extension allows user to extend the the set of intruments
 simply by dropping 1 or more  .wav files into the "InstrumentSamples" folder.

Single sampling and multi-sampling are both supported. 

For single sampling, put a single .wav file into "InstrumentSamples". 
The file name without extension will be used as the name of the instrument class.

For multisampling, first create a folder, whose name will be used as the name of the 
instrument class, then put multiple .wav files into the new created folder. The audio 
samples should span a range of different pitches. The sampler will generate notes by
intepolating between the samplers according to the target pitch.

The .wav file must have 1 or 2 channels in 16bit PCM format. 

The algorihm of the instrument sampler is by simply stretching the sample audio and
adding a envolope. So be sure the samples have sufficient length.


## Percussion Play

For percussion play, first you should consider what percussions to choose to build a
percussion group.
For example, I choose BassDrum and Snare:

```Python

	BassDrum=ScoreDraft.BassDrum()
	Snare=ScoreDraft.Snare()
	
	perc_list= [BassDrum, Snare]

```

Then, define 2 utility functions to refer to the noise generated by the 2 percussions:

```Python

	def dong(duration=48):
		return (0,duration)
	
	def ca(duration=48):
		return (1,duration)

```

Now you can use the above 2 functions to build a beat sequence like:

```Python

	seq = [dong(), ca(24), dong(24), dong(), ca(), dong(), ca(24), dong(24), dong(), ca()]

```

Create a document, play the beats and mixdown:

```Python

	doc=ScoreDraft.Document()
	doc.setTempo(120)
	doc.playBeatSeq(seq, perc_list)
	doc.mixDown('test_perc.wav')

```

### Percussion Sampler

The percussion sampler extension allows user to extend the the set of percussions simply by drop 1 .wav file
into the "PercussionSamples" folder.
The file name without extension will be used as the name of the percussion class.

The .wav file must have 1 or 2 channels in 16bit PCM format. 

The algorihm of the percussion sampler is by simply adding a envolope.
So be sure the samples have sufficient length.


## Singing

ScoreDraft provides a singing interface similar to instrument and percussion play.

A singing sequence is a little more complicated than a note-sequence. For example:

```Python

	seq= [ ("jin_f1", do(5,24), ti(4,24), do(5,24)), ("ke_f1", re(5,24)), ("la_f1", mi(5,24)) ]

```

Each singing segment contains one or more lyric as a string, each followed by 1 or more notes.

In many cases, there is only 1 note following a lyric. When there are more than 1 notes 
follwoing a lyric, all these notes will split the duration of that lyric. All lyrics and notes in the same tuple are intended to be sung continuously. However, when there are silence notes/backapsaces, the singing-segment will be broken into multiple segments to sing.


Create a document, sing the sequence and mixdown:

```Python

	doc=ScoreDraft.Document()
	doc.sing(seq, ScoreDraft.KeLaTest())
	doc.mixDown('KeLa.wav')

```

You can also mix raw notes without lyric with the singing segments. In that case, these 
notes will be sung using a default lyric.
Vice-versa, if you try playing a singing sequence with an instrument, the notes in the sequence will get played ignoring the lyrics.

### KeLa Engine

The KeLa engine is a simple singing engine provided as an extension. It synthesizes by 
streching individual audio samples into individual notes, which is similar to 単独音 in
UTAU. 

The audio samples used by KeLa Engine should be put into the "KeLaSamples" folder. Each
subfolder of "KeLaSamples" defines a singer class.

The sub-folder names are used as the name of the singer classes (like the "KeLaTest"
above). Each sub-folders contains multiple .wav files. The file names without 
extension are corresponding to the lyric strings in the singing segments.

Unlike the instrument sampler and the percussion sampler, KeLa engine takes in 
short pieces of audio samples, extract features and use the features to generate
 voice. So you don't need to use long audio samples, just try to sing a flat pitch 
during recording.

### UtauDraft Engine

The UtauDraft Enigine tries to be compatible with all kinds of UTAU voice-banks, including 単独音,連続音, VCV, CVVC as much as possible. oto.ini and .frq files will be used to understand the audio samples. prefix.map will also be used when one is present.

You can put a UTAU voice-bank directly into the "UTAUVoice" folder.  Each
subfolder of "UTAUVoice" defines a singer class.

The sub-folder names with "_UTAU" endings are used as the name of the singer classes. For 
example, the sub-folder "GePing" will define a singer class named "GePing_UTAU". It
the original sub-foler name is unsuitable to be used as an Python variable name, then you
should rename it to prevent a Python error.

When using UtauDraft Engine, you should use the names defined in oto.ini as lyrics, just 
like in UTAU. You can use 
"ScoreDraft.UtauDraftSetLyricConverter(singer, LyricConverterFunc)"
to set a converter function as a callback in order to convert the input lyrics to VCV or CVVC lyrics. For example:

```Python

	import ScoreDraft
	import CVVCChineseConverter

	Ayaka = ScoreDraft.Ayaka_UTAU()
	ScoreDraft.UtauDraftSetLyricConverter(Ayaka, CVVCChineseConverter.CVVCChineseConverter)

```

The converter function should have the following form:

```Python

	def LyricConverterFunc(LyricForEachSyllable):
		...
		return [(lyric1ForSyllable1, weight11, lyric2ForSyllable1, weight21...  ),(lyric1ForSyllable2, weight12, lyric2ForSyllable2, weight22...), ...]

```

The argument 'LyricForEachSyllable' has the form [lyric1, lyric2, ...], where each lyric is a string, which is the input lyric of a syllable.

The converter function should convert 1 input lyric into 1 or more lyrics to split the duration of the original syllable. A weight value should be provided to indicate the ratio
or duration of the converted note.

Examples of converter functions can be found in CVVCChineseConverter.py, TsuroVCVConverter.py and JPVCVConverter.py.

