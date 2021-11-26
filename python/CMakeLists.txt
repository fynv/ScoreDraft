cmake_minimum_required (VERSION 3.0)

set(PYTHON
ScoreDraft/__init__.py
ScoreDraft/ScoreDraftCore.py
ScoreDraft/Catalog.py
ScoreDraft/Initializers.py

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

ScoreDraft/MusicXMLDocument.py
ScoreDraft/YAMLDocument.py
)

set(PYTHON_MX
ScoreDraft/musicxml/__init__.py
ScoreDraft/musicxml/musicxml.py
ScoreDraft/musicxml/xlink.py
ScoreDraft/musicxml/xml.py
)

install(FILES ${PYTHON} DESTINATION ScoreDraft)
install(FILES ${PYTHON_MX} DESTINATION ScoreDraft/musicxml)
install(FILES setup.py README.md DESTINATION .)

