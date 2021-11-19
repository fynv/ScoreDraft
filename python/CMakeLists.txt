cmake_minimum_required (VERSION 3.0)

set(PYTHON
ScoreDraft/__init__.py
ScoreDraft/ScoreDraftCore.py

ScoreDraft/Instrument.py
ScoreDraft/Percussion.py
ScoreDraft/Singer.py
ScoreDraft/SimpleInstruments.py
ScoreDraft/KarplusStrong.py
ScoreDraft/BasicSamplers.py
ScoreDraft/SoundFont2.py
ScoreDraft/VoiceSampler.py
ScoreDraft/UTAUUtils.py
ScoreDraft/UtauDraft.py

ScoreDraft/RapChinese.py
ScoreDraft/CVVCChineseConverter.py
ScoreDraft/XiaYYConverter.py
ScoreDraft/JPVCVConverter.py
ScoreDraft/TsuroVCVConverter.py
ScoreDraft/TTEnglishConverter.py
ScoreDraft/VCCVEnglishConverter.py
ScoreDraft/TTLyricSet.data
ScoreDraft/VCCVLyricSet.data

ScoreDraft/Document.py
ScoreDraft/Notes.py
ScoreDraft/MIDIWriter.py
ScoreDraft/PCMPlayer.py
ScoreDraft/Meteor.py

ScoreDraft/Catalog.py
)

install(FILES ${PYTHON} DESTINATION ScoreDraft)
install(FILES setup.py README.md DESTINATION .)

