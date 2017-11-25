#include "WinWavWriter.h"
#include "TrackBuffer.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

struct WavHeader
{
	unsigned short wFormatTag;
	unsigned short wChannels;
	unsigned int dwSamplesPerSec;
	unsigned int dwAvgBytesPerSec;
	unsigned short wBlockAlign;
	unsigned short wBitsPerSample;
}; 

void WriteToWav(TrackBuffer& track, const char* fileName)
{
	unsigned numSamples = track.NumberOfSamples();
	unsigned sampleRate = track.Rate();
	float volume = track.Volume();

	unsigned dataSize = numSamples * 2;
	unsigned int adWord;
	WavHeader header;

	FILE* fp = fopen(fileName, "wb");
	fwrite("RIFF", 1, 4, fp);
	adWord = dataSize + 8 + 4 + sizeof(WavHeader);
	fwrite(&adWord, 4, 1, fp);
	fwrite("WAVEfmt ", 1, 8, fp);
	adWord = 0x00000010;
	fwrite(&adWord, 4, 1, fp);

	header.wFormatTag = 1;
	header.wChannels = 1;
	header.dwSamplesPerSec = sampleRate;
	header.dwAvgBytesPerSec = sampleRate * sizeof(short);
	header.wBlockAlign = 2;
	header.wBitsPerSample = 16;

	fwrite(&header, sizeof(WavHeader), 1, fp);
	fwrite("data", 1, 4, fp);
	adWord = dataSize;
	fwrite(&adWord, 4, 1, fp);

	unsigned bufferSize = numSamples;
	short* data = new short[bufferSize];

	unsigned i;
	for (i = 0; i<bufferSize; i++)
		data[i] = (short)(max(min(track.Sample(i)*volume, 1.0f), -1.0f)*32767.0f);

	fwrite(data, sizeof(short), bufferSize, fp);

	delete[] data;

	fclose(fp);
}
