#include "WriteWav.h"

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

WriteWav::WriteWav()
{
	m_fp = nullptr;
}

WriteWav::~WriteWav()
{
	if (m_fp) fclose(m_fp);
}

bool WriteWav::OpenFile(const char* filename)
{
	if (m_fp) fclose(m_fp);
	m_fp = fopen(filename, "wb");
	return m_fp != nullptr;
}

void WriteWav::CloseFile()
{
	if (m_fp) fclose(m_fp);
	m_fp = nullptr;
}

void WriteWav::WriteHeader(unsigned sampleRate, unsigned numSamples, unsigned chn)
{
	if (!m_fp) return;
	unsigned dataSize = numSamples * chn * sizeof(short);
	unsigned int adWord;
	WavHeader header;

	fwrite("RIFF", 1, 4, m_fp);
	adWord = dataSize + 20 + sizeof(WavHeader);
	fwrite(&adWord, 4, 1, m_fp);
	fwrite("WAVEfmt ", 1, 8, m_fp);
	adWord = 0x00000010;
	fwrite(&adWord, 4, 1, m_fp);

	header.wFormatTag = 1;
	header.wChannels = chn;
	header.dwSamplesPerSec = sampleRate;
	header.dwAvgBytesPerSec = sampleRate *chn* sizeof(short);
	header.wBlockAlign = chn* sizeof(short);
	header.wBitsPerSample = 16;

	fwrite(&header, sizeof(WavHeader), 1, m_fp);
	fwrite("data", 1, 4, m_fp);
	adWord = dataSize;
	fwrite(&adWord, 4, 1, m_fp);

	m_totalSamples = numSamples;
	m_num_channels = chn;
	m_writenSamples = 0;
}

void WriteWav::WriteSamples(const float* samples, unsigned count, float volume)
{
	if (!m_fp) return;
	count = min(count, m_totalSamples - m_writenSamples);
	if (count > 0)
	{
		short* data = new short[count*m_num_channels];

		unsigned i;
		for (i = 0; i<count*m_num_channels; i++)
			data[i] = (short)(max(min(samples[i]*volume, 1.0f), -1.0f)*32767.0f);

		fwrite(data, sizeof(short), count*m_num_channels, m_fp);

		delete[] data;

		m_writenSamples += count;
	}
	if (m_totalSamples - m_writenSamples<=0) CloseFile();
}

