#ifndef _WriteWav_h
#define _WriteWav_h

#include <stdio.h>
class WriteWav
{
public:
	WriteWav();
	~WriteWav();

	bool OpenFile(const char* filename);
	void CloseFile();

	void WriteHeader(unsigned sampleRate, unsigned numSamples, unsigned chn=1);
	void WriteSamples(const float* samples, unsigned count, float volume);

private:
	FILE* m_fp;
	unsigned m_totalSamples;
	unsigned m_num_channels;
	unsigned m_writenSamples;
};

#endif
