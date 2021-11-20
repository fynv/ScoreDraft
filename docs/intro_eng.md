# Introduction

The soure-code of ScoreDraft is hosted on [GitHub](https://github.com/fynv/ScoreDraft), where you can always find the latest changes that I have made.

PyPi packages for Windows x64 & Linux x64 are available for download by

```
pip install scoredraft
```

This document will introduce the uses of each basic elements of ScoreDraft.

## HelloWorld Example

Let's start from a minimal example to explain the basic usage and design ideas of ScoreDraft.


```python
import ScoreDraft
from ScoreDraft.Notes import *

seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

buf=ScoreDraft.TrackBuffer()
ScoreDraft.KarplusStrongInstrument().play(buf, seq)
ScoreDraft.WriteTrackBufferToWav(buf,'twinkle.wav')
```

<audio controls>
	<source type="audio/mpeg" src="twinkle.mp3"/>
</audio>

### Play Calls