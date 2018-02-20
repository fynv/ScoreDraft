import sys
sys.path+=['../']

import ScoreDraft
from tts import TTS

buf=TTS("光头")
ScoreDraft.WriteTrackBufferToWav(buf,'test.wav')
