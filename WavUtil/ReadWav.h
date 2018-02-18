#ifndef _ReadWav_h
#define _ReadWav_h

#include <stdio.h>
class ReadWav
{
public:
	ReadWav();
	~ReadWav();

	bool OpenFile(const char* filename);
	void CloseFile();

	bool ReadHeader(unsigned &sampleRate, unsigned &numSamples, unsigned& chn);
	bool ReadSamples(float* samples, unsigned count, float& maxv);

private:
	FILE* m_fp;
	unsigned m_totalSamples;
	unsigned m_num_channels;
	unsigned m_readSamples;
};

#endif
