#include "InstrumentSample.h"
#include <ReadWav.h>
#include "FrequencyDetection.h"
#include <string.h>
#include <string>

InstrumentSample::InstrumentSample()
{
	m_wav_length = 0;
	m_wav_samples = nullptr;
}

InstrumentSample::~InstrumentSample()
{
	delete[] m_wav_samples;
}

bool InstrumentSample::LoadWav(const char* wav_path)
{
	delete[] m_wav_samples;
	m_wav_length = 0;
	m_chn = 1;
	m_wav_samples = nullptr;

	ReadWav reader;
	reader.OpenFile(wav_path);
	if (!reader.ReadHeader(m_origin_sample_rate, m_wav_length, m_chn)) return false;

	m_wav_samples = new float[m_wav_length*m_chn];
	if (!reader.ReadSamples(m_wav_samples, m_wav_length, m_max_v))
	{
		delete[] m_wav_samples;
		return false;
	}

	_fetchOriginFreq(wav_path);

	return true;
}

void InstrumentSample::_fetchOriginFreq(const char* wav_path)
{
	std::string freq_path = std::string(wav_path).substr(0, strlen(wav_path) - 4) + ".freq";
	FILE *fp = fopen(freq_path.data(), "r");
	if (fp)
	{
		fscanf(fp, "%f", &m_origin_freq);
		fclose(fp);
	}
	else
	{
		float* localMono = nullptr;
		float* pSamples = nullptr;
		if (m_chn == 1)
		{
			pSamples = m_wav_samples;
		}
		else if (m_chn == 2)
		{
			float* localMono = new float[m_wav_length];
			pSamples = localMono;
			for (unsigned i = 0; i < m_wav_length; i++)
			{
				localMono[i] = 0.5f*(m_wav_samples[i * 2] + m_wav_samples[i * 2 + 1]);
			}
		}
		m_origin_freq = fetchFrequency(m_wav_length, pSamples, m_origin_sample_rate);
		printf("Detected frequency of %s : %fHz\n", wav_path, m_origin_freq);
		fp = fopen(freq_path.data(), "w");
		fprintf(fp, "%f\n", m_origin_freq);
		fclose(fp);

		delete[] localMono;
	}
}

