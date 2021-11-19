#include "ReadWav.h"
#include <cmath>

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


ReadWav::ReadWav()
{
	m_fp = nullptr;
}

ReadWav::~ReadWav()
{
	if (m_fp) fclose(m_fp);
}

bool ReadWav::OpenFile(const char* filename)
{
	if (m_fp) fclose(m_fp);
	m_fp = fopen(filename, "rb");
	return (m_fp != nullptr);
}

void ReadWav::CloseFile()
{
	if (m_fp) fclose(m_fp);
	m_fp = nullptr;
}


bool ReadWav::ReadHeader(unsigned& sampleRate, unsigned& numSamples, unsigned& chn)
{
	if (!m_fp) return false;

	char c_riff[4] = { 'R', 'I', 'F', 'F' };
	unsigned& u_riff = *(unsigned*)c_riff;

	char c_wave[4] = { 'W', 'A', 'V', 'E' };
	unsigned& u_wave = *(unsigned*)c_wave;

	char c_fmt[4] = { 'f', 'm', 't', ' ' };
	unsigned& u_fmt = *(unsigned*)c_fmt;

	char c_data[4] = { 'd', 'a', 't', 'a' };
	unsigned& u_data = *(unsigned*)c_data;

	unsigned buf32;

	fread(&buf32, 4, 1, m_fp);
	if (buf32 != u_riff)
	{
		fclose(m_fp);
		return false;
	}

	unsigned chunkSize;
	fread(&chunkSize, 4, 1, m_fp);

	fread(&buf32, 4, 1, m_fp);
	if (buf32 != u_wave)
	{
		fclose(m_fp);
		return false;
	}

	fread(&buf32, 4, 1, m_fp);
	if (buf32 != u_fmt)
	{
		fclose(m_fp);
		return false;
	}

	unsigned headerSize;
	unsigned skipSize = 0;
	unsigned skipBuffer;

	fread(&headerSize, 4, 1, m_fp);
	if (headerSize < sizeof(WavHeader))
	{
		fclose(m_fp);
		return false;
	}
	else if (headerSize > sizeof(WavHeader))
	{
		if (headerSize - sizeof(WavHeader) > 4)
		{
			fclose(m_fp);
			return false;
		}
		skipSize = headerSize - sizeof(WavHeader);
	}

	WavHeader header;
	fread(&header, sizeof(WavHeader), 1, m_fp);

	if (skipSize > 0)
	{
		fread(&skipBuffer, 1, skipSize, m_fp);
	}

	if (header.wFormatTag != 1)
	{
		fclose(m_fp);
		return false;
	}

	chn = header.wChannels;
	if (chn<1 || chn>2)
	{
		fclose(m_fp);
		return false;
	}

	sampleRate = header.dwSamplesPerSec;

	if (header.wBitsPerSample != 16)
	{
		fclose(m_fp);
		return false;
	}

	fread(&buf32, 4, 1, m_fp);
	if (buf32 != u_data)
	{
		fclose(m_fp);
		return false;
	}

	unsigned dataSize;
	fread(&dataSize, 4, 1, m_fp);

	numSamples = dataSize / chn / 2;

	m_totalSamples = numSamples;
	m_num_channels = chn;
	m_readSamples = 0;

	return true;
}


bool ReadWav::ReadSamples(float* samples, unsigned count, float& max_v)
{
	if (!m_fp) return false;
	count = min(count, m_totalSamples - m_readSamples);

	if (count > 0)
	{
		short* data = new short[count*m_num_channels];
		fread(data, sizeof(short), count*m_num_channels, m_fp);

		max_v = 0.0f;
		for (unsigned i = 0; i < count*m_num_channels; i++)
		{	
			float v = (float)data[i] / 32767.0f;
			samples[i] = v;
			max_v = max(max_v, fabsf(v));
		}
		delete[] data;

		m_readSamples += count;
	}

	if (m_totalSamples - m_readSamples <= 0) CloseFile();

	return true;
}
