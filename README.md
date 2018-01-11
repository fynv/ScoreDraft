Warning: Major changes are being made to the PyScoreDraft interface. Document will be updated soon.
================

ScoreDraft
================
A simple software music synthesizer and text/script based score authoring. Used for music experiments. 

In the first design, I was intending to use text files as the means of authoring, i.e., the C++ program
reads a text file, parse the notes and generates a music file. However, after trying using Python as a 
script layer, it has shown very much benefit comparing to text parsing. 

Now, Python is the recommended way to use ScoreDraft. The following examples shows how easily a piece of
music can be sythesized using the Python interface of ScoreDraft. 


```Python
import ScoreDraft
from ScoreDraftNotes import *

doc=ScoreDraft.Document()

seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

doc.playNoteSeq(seq, ScoreDraft.Piano())
doc.mixDown('twinkle.wav')

```

# Using ScoreDraft.py interface

## Note-sequences

A list like the following defines a "note-sequence":

[ (freq, duration), (freq, duration), (freq, duration)... ]

A note-sequence can by "played" using an "instrument".

Each tuple (freq, duration) defines a "note". 

* "freq" is a floating point frequency multiplier relative to a reference frequency.
* "duration" is a integer that defines the duration of the note. 1 beat = 48, 1/2 beat =24 etc.

"freq" is typically a positive number. A negative "freq" number has the following special usage:
* when "freq" is negative and "duration" is positive, a silent time period is defined, which 
  occupies the "duration" but does not make any sound.
* when "freq" is negative and "duration" is also negative, a "backspace" is generated. The cursor
  of note-sequence play will be brought backwards by "duration". In this way, the new notes can
  be overlapped with old ones, so that harmonies and chords can be generated.

The utility file "ScoreDraftNotes.py" contains utility functions do(), re(), mi() etc., which 
simplies the frequency calculation. For example, in "do(octave, duration)", octave defines the 
octave of the "do" sound, where octave=5 means the central octave, and do(5,48) defines a 1 beat
sound at the reference frequency. The return value is a tuple with the proper "freq" calculated.
BL() generates a silent time period and BK() generates a backspace.

## Beat-sequences

A list like the following defines a "beat-sequence":

[ (index, duration), (index, duration), (index, duration)... ]

A beat-sequence can by "played" using a list of "percussions".

Each tuple (index, duration) defines a "beat". 

* "index" is a integer. When it is non-negative, it references a percussion in a percussion list, which is used to make to noise
* "duration" is the same as the "duration" of a note

Negative "index" values are used in the same way as negative "freq" values of notes to define
silent time periods and backspaces.

## Tuning commands

In both note-sequences and beat-sequences, tuning commands can be inserted to tune an instrument or a percussion. 
A tuning command is a string. Different instruments and percussions can support different tuning commands.
Currently, the only supported tuning command is 'volume [x]'. Examples:

* in a note sequence: [do(5,48),re(5,48), "volume 2.0", mi(5,48)... ]
* in a beat sequence: [(0,48), (1,48), (0,"volume 2.0"), (1,"volume 3.0"), (0,48), (1,48)... ]

In the beat sequence case, an index need to be provided to choose which persecussion the command is sent to.

## ScoreDraft.Document

"ScoreDraft.Document" is a class used to manage some public states like tempo and reference-frequency.
It also maintains a list of "track-buffers", which are generated wave forms.

* ScoreDraft.Document.tempo is an integer defining the tempo in beats/minute.
* ScoreDraft.Document.refFreq is a floating point defining the reference-frequency in Hz.

* ScoreDraft.Document.playNoteSeq(self, seq, instrument, volume): 
  - seq is the note-sequence to play
  - instrument is the instrument used to play the sequence. The usage of instrument will be explained later.
  - volume is a floating point to define volume of the track. It is applied during mixing, and has nothing to do with the instrument tunning. 
  A track-buffer will be generated and recorded.

* ScoreDraft.Document.playBeatSeq(self, seq, percList, volume):
  - seq is the beat-sequence to play
  - percList is a list of percussions used to play the sequence. The usage of percussion will be explained later.
  - volume is a floating point to define volume of the track. It is applied during mixing, and has nothing to do with the percussion tunning. 
  A track-buffer will be generated and recorded.

* ScoreDraft.Document.trackToWav(self,trackIndex, filename):
  - trackIndex is an integer, indicating a track, according to the call sequence of playNoteSeq(), playBeatSeq() etc.
  - filename is a string. 
  A windows wav file with the given name will be generated containing a specified track.

* ScoreDraft.Document.mixDown(self,filename):
  - filename is a string. 
  A windows wav file with the given name will be generated by mixing down all the generated tracks.

## MIDI write-down

ScoreDraft.WriteNoteSequencesToMidi(noteSeqList, tempo, refFreq, filename)
is an interface to write note-sequences to a MIDI file.

* noteSeqList is a list to note-sequences. Each note-sequence will be converted to a MIDI track.
* tempo is an integer defining the tempo in beats/minute.
* refFreq is a floating point defining the reference-frequency in Hz.
* filename is a string. 
A MIDI file with the given name will be generated.

## Instrument/Percussion instancing

ScoreDraft.g_instList contains a list of all available instruments. Run print(ScoreDraft.g_instList)
to display the list, we will see something like:

['PureSin', 'Square', 'Triangle', 'Sawtooth', 'NaivePiano', 'BottleBlow', 'Ah', 'Cello', 'Lah', 'Piano', 'String', 'Violin']

A function corresponding to each of the above names will be generated, such as:

ScoreDraft.PureSin(), ScoreDraft.Square()..

They can be used to create instances of the instruments.

Similarly, ScoreDraft.g_percList contains a list of all available percussions, and there are 
functions to create instances of percussions like:

ScoreDraft.BassDrum(), ScoreDraft.Snare()..

# Sampling supports

Sampling based synthesis is now supported using 2 extensions, which supports instrument sampling
and percussion sampling separately.

The "InstrumentSamples" directory is searched for .wav files. Each file is used to define an 
instrument. For example, if you put a "whistle.wav" into the "InstrumentSamples" directory, 
an instrument named "whistle" will be available to use.

Similary, .wav files in the "PercussionSamples" direcory are used to define percussions. If you
put a "clap.wav" into the "PercussionSamples" directory, an percussion named "clap" will be
available to use.

# Introduction to the PyScoreDraft.pyd interface

PyScoreDraft.pyd is the C++ based core module underlying ScoreDraft.py.
The following interfaces are provided:

* PyScoreDraft.ListInstruments()
  - returns a string list of all available instruments

* PyScoreDraft.InitInstrument(clsId)
  - clsId: an integer referencing an instrument class in the list returned by ListInstruments()
  - returns an id of the initialized instrument instance

* PyScoreDraft.ListPercussions()
  - returns a string list of all available percussions

* PyScoreDraft.InitPercussion(clsId)
  - clsId: an integer referencing an percussion class in the list returned by ListPercussions()
  - return an id of the initialized percussion instance

* PyScoreDraft.InstrumentPlay(InstrumentId, seq, volume, tempo, RefFreq)
  - InstrumentId: an integer, id of the instrument instance, which is return by InitInstrument()
  - seq: the note-sequence to play
  - volume: a floating point defining the volume of the track
  - tempo: an integer defining the tempo in beats/minute.
  - RefFreq: a floating point defining the reference-frequency in Hz.
  - return an id of the generated track-buffer

* PyScoreDraft.InstrumentTune(InstrumentId, cmd)
  - InstrumentId: an integer, id of the instrument instance, which is return by InitInstrument()
  - cmd: a string to be passed to the instrument

* PyScoreDraft.PercussionPlay(percIdList, seq, volume, tempo)
  - percIdList: a list of percussion instance ids, each id is returned by InitPercussion()
  - seq: the beat-sequence to play
  - volume: a floating point defining the volume of the track
  - tempo: an integer defining the tempo in beats/minute.
  - return an id of the generated track-buffer

* PyScoreDraft.PercussionTune(PercussionId, cmd)
  - PercussionId: an integer, id of the percussion instance, which is return by InitPercussion()
  - cmd: a string to be passed to the percussion

* PyScoreDraft.MixTrackBufferList(trackBufferIdList)
  - trackBufferIdList: a list of track-buffer ids, each id is returned by InstrumentPlay() or PercussionPlay() or MixTrackBufferList()
  - return a new id of the mixed track-buffer

* PyScoreDraft.WriteTrackBufferToWav(trackBufferId, filename)
  - trackBufferId: the id of the track-buffer to write, returned by InstrumentPlay() or PercussionPlay() or MixTrackBufferList()
  - filename: the name of the wav file

* PyScoreDraft.WriteNoteSequencesToMidi(noteSeqList, temp, refFreq, filename)
  - noteSeqList is a list to note-sequences. Each note-sequence will be converted to a MIDI track.
  - tempo is an integer defining the tempo in beats/minute.
  - refFreq is a floating point defining the reference-frequency in Hz.
  - filename is a string. 

# Extension interface 

The set of instruments and percussions can be extended by separate dll builds.
The dll files are searched by PyScoreDraft.pyd within "Extensions" directory.

Each of these dll files should provide an interface funcion:

```C++
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C" __declspec(dllexport) InstrumentFactory*
PY_SCOREDRAFT_EXTENSION_INTERFACE GetFactory();
```

The class "InstrumentFactory" is defined in "PyScoreDraft.h". Implementation examples can be found
in "PyScoreDraft.cpp" (TypicalInstrumentFactory), "InstrumentSampler.cpp" (InstrumentSamplerFactory),
and "PercussionSampler.cpp" (PercussionSamplerFactory).
