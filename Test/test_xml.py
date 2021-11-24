import ScoreDraft as sd

# doc = sd.from_music_xml("ZhenDeAiNi.xml")
doc = sd.from_lilypond("ZhenDeAiNi.ly")

instrument = sd.SF2Instrument('florestan-subset.sf2', 0)
# instrument = sd.Sawtooth()

doc.playXML([instrument])
# doc.mixDown('ZhenDeAiNi.wav')
doc.meteor()

