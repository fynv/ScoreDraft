cmake_minimum_required (VERSION 3.0)

set(PYTHON
__init__.py
Extensions.py
TrackBuffer.py
Instrument.py
Percussion.py
Singer.py
Document.py
InternalInstruments.py
Notes.py
RapChinese.py
PercussionSampler.py
InstrumentSampler.py
KeLa.py
UtauDraft.py
Meteor.py
SF2Instrument.py
KarplusStrongInstrument.py

CVVCChineseConverter.py
JPVCVConverter.py
TsuroVCVConverter.py
TTEnglishConverter.py
TTLyricSet.data
VCCVEnglishConverter.py
VCCVLyricSet.data
XiaYYConverter.py

Catalog.py
)

install(FILES ${PYTHON} DESTINATION ./ScoreDraft)
