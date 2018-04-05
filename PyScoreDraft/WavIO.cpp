#include "WavIO.h"
#include <WriteWav.h>
#include <ReadWav.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

void WriteToWav(TrackBuffer& track, const char* fileName)
{
	unsigned numSamples = track.NumberOfSamples();
	unsigned chn = track.NumberOfChannels();
	unsigned sampleRate = track.Rate();
	float volume = track.AbsoluteVolume();
	float pan = track.Pan();

	WriteWav writer;
	writer.OpenFile(fileName);
	writer.WriteHeader(sampleRate, numSamples, chn);

	unsigned localBufferSize = track.GetLocalBufferSize();
	float *buffer = new float[localBufferSize*chn];
	unsigned pos = 0;
	while (numSamples > 0)
	{
		unsigned writeCount = min(numSamples, localBufferSize);
		track.GetSamples(pos, writeCount, buffer);
		writer.WriteSamples(buffer, writeCount, volume, pan);
		numSamples -= writeCount;
		pos += writeCount;
	}

	delete[] buffer;
}

void ReadFromWav(TrackBuffer& track, const char* fileName)
{
	unsigned numSamples;
	unsigned chn;
	unsigned sampleRate;

	ReadWav reader;
	reader.OpenFile(fileName);
	reader.ReadHeader(sampleRate, numSamples, chn);

	unsigned localBufferSize = track.GetLocalBufferSize();

	NoteBuffer buf;
	buf.m_sampleRate = sampleRate;
	buf.m_channelNum = chn;
	buf.m_sampleNum = localBufferSize;
	buf.Allocate();

	while (numSamples > 0)
	{
		unsigned readCount = min(numSamples, localBufferSize);
		float maxv;
		reader.ReadSamples(buf.m_data, readCount, maxv);
		buf.m_sampleNum = readCount;
		buf.m_cursorDelta = (float)readCount;
		track.WriteBlend(buf);
		numSamples -= readCount;
	}
}
