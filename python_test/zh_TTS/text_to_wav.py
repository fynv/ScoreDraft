import sys
sys.path+=['../']

import ScoreDraft
from tts import TTS

buf=ScoreDraft.TrackBuffer(1)
TTS("光头",buf)
ScoreDraft.WriteTrackBufferToWav(buf,'test.wav')
