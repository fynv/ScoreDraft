ScoreDraft
================
A simple software music synthesizer and Python based score authoring. 

Historically, there has been some changes in the design goal of the software.
In the beginning, the software was intendend to be used for music experiments like just intonation,
and in the first design,  I was intending to use text files as the means of authoring, i.e., the C++ program
reads a text file, parse the notes and generates an audio file. However, after trying using Python as a 
script layer, it has shown very much benefit comparing to text parsing. So I moved quickly to Python.

Then, I became interesting in extending ScoreDraft for singing sythesis, and it has become the focus 
of the developent of ScoreDraft recently. Protocols for singing sythesis are added, and basic resampling 
functionality similar to Utau has been implemented.

I'm really not sure where ScoreDraft will go in the furture. Basically, here, I'm building a big set
of utilities to allow different kinds all waveforms to be generated and mixed together through scripts. 

The following example shows how easily a piece of musical sound can be generated using ScoreDraft.


```Python
import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()

seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

doc.playNoteSeq(seq, ScoreDraft.Piano())
doc.mixDown('twinkle.wav')

```

# Repository contents

Python layer and the binaries built from C++ are kept in the "python_test" directory.
These files are being dynamically updated as the development progresses. 
Python space users who uses Windows-x64 or Linux-amd64 only need this directory to use ScoreDraft.
Audio resource files will be kept at small scale here. Users can extend that part themselves.

The rest of the repository are all source-code managed with CMake.

I have no plan to put release packages on GitHub. 
There might be some special packaging released through forums in the future, which may
contain special audio resources for demoing.

# Usage

First
```Python
 import ScoreDraft
```

Python space interfaces are defined and documented in python_test/ScoreDraft.py.
Here I only give a brief introduction by examples. And, forget about the PyScoreDraft interface.
All of the interfaces are properly wrapped by ScoreDraft.py. You won't need it unless 
you are developing an extension of PyScoreDraft/ScoreDraft in C++.

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

For instrument play, write you note sequence like:
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

## Instrument Sampler

The instrument sampler extension allows user to extend the the set of intruments simply by drop 1 or more 
.wav files into the "InstrumentSamples" folder.

Single sampling and multi-sampling are both supported. 

For single sampling, put a single .wav file into "InstrumentSamples". 
The file name without extension will be used as the name of the instrument class.

For multisampling, first create a folder, whose name will be used as the name of the instrument class,
then put multiple .wav files into the new created folder. The audio samples should span a range of 
different pitches. The sampler will generate notes by intepolating between the samplers according to 
the target pitch.

The .wav file must have 1 or 2 channels in 16bit PCM format. 

The algorihm of the instrument sampler is by simply stretching the sample audio and adding a envolope.
So be sure the samples have sufficient length.


## Percussion Play

For percussion play, first you should consider what percussions to choose to build a percussion group.
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

## Percussion Sampler

The percussion sampler extension allows user to extend the the set of percussions simply by drop 1 .wav file
into the "PercussionSamples" folder.
The file name without extension will be used as the name of the percussion class.

The .wav file must have 1 or 2 channels in 16bit PCM format. 

The algorihm of the percussion sampler is by simply adding a envolope.
So be sure the samples have sufficient length.


## Singing and KeLa

ScoreDraft provides a singing interface similar to instrument and percussion play.
Currently, all "singers" are provided by the KeLa module, a simple voice resampler.

A singing sequence is a little more complicated than a note-sequence. For example:
```Python
  seq= [ ("jin_f1", do(5,24), ti(4,24), do(5,24)), ("ke_f1", re(5,24)), ("la_f1", mi(5,24)) ]
```

Each singing segment contains a lyric as a string, followed by 1 or more notes.
When there are more than 1 notes in a singing segment, the notes will be sung continuously.

Create a document, sing the sequence and mixdown:

```Python
  doc=ScoreDraft.Document()
  doc.sing(seq, ScoreDraft.KeLaTest())
  doc.mixDown('KeLa.wav')
```

You can also mix raw notes without lyric with the singing segments. In that case, these 
notes will be sung using a default lyric.
Vice-versa, if you try playing a singing sequence with an instrument, the notes in the sequence will get played ignoring the lyrics.

The KeLa module synthesizes singing sound using samples in the "KeLaSamples" folder.
Each subfolder of "KeLaSamples" defines a singer class.
The sub-folder names are used as the name of the singer classes (like the "KeLaTest" above).
Each sub-folders contains multiple .wav files. The file names without extension are 
corresponding to the lyric strings in the singing segments.

Unlike the instrument sampler and the percussion sampler, KeLa engine takes in short pieces of audio samples, 
extract features and use the features to generate voice. So you don't need to use long audio samples, just 
try to sing a flat pitch during recording.
