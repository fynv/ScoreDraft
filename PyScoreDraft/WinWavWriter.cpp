#include "WinWavWriter.h"
#include "TrackBuffer.h"
#include <WriteWav.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

void WriteToWav(TrackBuffer& track, const char* fileName)
{
	unsigned numSamples = track.NumberOfSamples();
	unsigned sampleRate = track.Rate();
	float volume = track.Volume();

	WriteWav writer;
	writer.OpenFile(fileName);
	writer.WriteHeader(sampleRate, numSamples);

	unsigned localBufferSize = track.GetLocalBufferSize();
	float *buffer = new float[localBufferSize];
	unsigned pos = 0;
	while (numSamples > 0)
	{
		unsigned writeCount = min(numSamples, localBufferSize);
		track.GetSamples(pos, writeCount, buffer);
		writer.WriteSamples(buffer, writeCount, volume);
		numSamples -= writeCount;
		pos += writeCount;
	}

	delete[] buffer;
}

