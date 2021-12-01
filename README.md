[-> 中文Readme](README_cn.md)

# ScoreDraft

ScoreDraft is a music/singing synthesizer that provides a Python based 
score authoring interface. 

Currently, it includes the following synthesizer engines:

* Instrumental
  - SimpleInstruments: simple mathemtatic functions like sine-waves
  - KarplusStrong: guitar simulator using pure algorithm
  - BasicSamplers: tones generated by sampling one or multiple waveform samples
  - SoundFont2
* Percussion
  - BasicSamplers
  - SoundFont2 (GM Drums)
* Voice
  - VoiceSampler: a PSOLA-like algorithm is used to sample voice samples. A front-end "UtauDraft" is provided making it compatible to existing UTAU voice banks.

The framework is open and can be easily extended.

A PCM Player is provided to playback and visualize generated waveforms.

A more sophisticated visualizer "Meteor" is provided to visualize the notes/beats/syllables where the music is generated from.

The following example shows how easily a piece of musical sound can be generated using ScoreDraft.

```Python
    import ScoreDraft
    from ScoreDraft.Notes import *

    doc=ScoreDraft.Document()

    seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

    doc.playNoteSeq(seq, ScoreDraft.Piano())
    doc.mixDown('twinkle.wav')
```

Starting from version 1.0.3, ScoreDraft now supports a YAML based input format, which looks like:

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

where in the 'content' part, LilyPond syntax can be used to input notes. The following shell command can be used to generate a wav:

```
# scoredraft -wav twinkle.wav test.yaml
```

For more detailed introduction and demos see: [https://fynv.github.io/ScoreDraft/](https://fynv.github.io/ScoreDraft/)

## Installation

ScoreDraft is now available from PyPi. Windows x64/Linux x64 supported.

```
# pip install scoredraft
```

Known issue: For Linux, it is only tested on Ubuntu 20.04. It is known to be not working on Ubuntu 18.04.

## Building

Build-time dependencies:

* CMake 3.0+

* Python3

* CUDA(optional): To build without CUDA, disable the CMake option "USE_CUDA"

* FreeType: 
  
  - Library included for Windows
  - Ubuntu: sudo apt install libfreetype-dev

* GLFW: 
  
  - Source code included
  - Ubuntu: sudo apt install libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev

* PortAudio:
  
  - Source code included
  - Ubuntu: sudo apt-get install libasound-dev libjack-dev

Build process:

```
# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=../Test
# make
# make install
```

Run-time dependencies:

* Python3 
* cffi
* X-org, ALSA drivers are optionally needed for players
* xsdata, python_ly are optionally needed for MusicXML and LilyPond support
* pyyaml is optionally needed for YAML support

## Samples & Voice Banks

To avoid causing troubles, ScoreDraft now only includes a minimal set of instrumental/percussion samples to support the tests. The PyPi package doesn't include any of these.

ScoreDraft indexes the samples and banks by searching specific directory where the python script is started.

* Directory InstrumentSamples: wav instrument samples
  
  - There can be sub-directories containing multiple wav samples defining a single instrument for different pitch ranges.
  - Optionally, a freq file can be provided to define the frequency of the sample tone.

* Directory PercussionSamples: wav percussion samples

* Directory SF2: SoundFont2 bank files

* Directory UTAUVoice: UTAU voice banks (as sub-directories)

Users need to help themselves to download and organize extra samples and voicebanks. Here are just some recommandations, which is what I do at my place.

* Instrument/Percussion wav samples:
  
  - https://freewavesamples.com

* SoundFont2 Instruments
  
  - Arachno: http://www.arachnosoft.com/main/download.php?id=soundfont-sf2
  - SynthFontViena: http://www.synthfont.com/

* UTAU
  
  - uta(Japanese): default voice-bank that comes with UTAU
  - TetoEng(English): https://kasaneteto.jp/en/voicebank.html
  - Ayaka(Chinese): https://bowlroll.net/file/53297
  - Ayaka2 (Japanese): https://bowlroll.net/file/69898
  - WanEr(Chinese): http://lindayana.lofter.com/waner

## License

ScoreDraft is now formally available under [MIT license](https://choosealicense.com/licenses/mit/).

## Version History

ScoreDraft was my first Python project. There were some severe limitations for pythonic packaging and releasing.
As a result, there was no formal release before the Nov 2021 refactoring (version 1.0.0).

[SingingGadgets](https://pypi.org/project/singinggadgets/) was a refactoring attampt in 2018, 
which only partially solves the issues. 

After the Nov 2021 refactoring, ScoreDraft have all the benefits of both SingingGadgets and the old ScoreDraft.
Therefore, the SingingGadgets project is now closed.

* Nov 29, 2021. ScoreDraft 1.0.4, 1.0.5, YAML function updates
* Nov 27, 2021. ScoreDraft 1.0.3, adding a YAML based input routine support
* Nov 24, 2021. ScoreDraft 1.0.2, adding MusicXML & LilyPond support
* Nov 19, 2021. ScoreDraft 1.0.0 & 1.0.1
* Jun 16, 2018. SingingGadgets 0.0.3
