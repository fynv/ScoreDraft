#include "InstrumentSample.h"
#include <ReadWav.h>
#include "FrequencyDetection.h"

InstrumentSample::InstrumentSample()
{
	m_wav_length = 0;
	m_wav_samples = nullptr;
}

InstrumentSample::~InstrumentSample()
{
	delete[] m_wav_samples;
}

bool InstrumentSample::LoadWav(const char* root, const char* name, const char* instrumentName)
{
	char filename[1024];
	if (instrumentName==nullptr)
		sprintf(filename, "%s/InstrumentSamples/%s.wav", root, name);
	else
		sprintf(filename, "%s/InstrumentSamples/%s/%s.wav", root, instrumentName, name);

	delete[] m_wav_samples;
	m_wav_length = 0;
	m_wav_samples = nullptr;

	ReadWav reader;
	reader.OpenFile(filename);
	if (!reader.ReadHeader(m_origin_sample_rate, m_wav_length)) return false;

	m_wav_samples = new float[m_wav_length];
	if (!reader.ReadSamples(m_wav_samples, m_wav_length, m_max_v))
	{
		delete[] m_wav_samples;
		return false;
	}

	_fetchOriginFreq(root, name, instrumentName);

	return true;
}

void InstrumentSample::_fetchOriginFreq(const char* root, const char* name, const char* instrumentName)
{
	char filename[1024];
	if (instrumentName == nullptr)
		sprintf(filename, "%s/InstrumentSamples/%s.freq", root, name);
	else
		sprintf(filename, "%s/InstrumentSamples/%s/%s.freq", root, instrumentName, name);

	FILE *fp = fopen(filename, "r");
	if (fp)
	{
		fscanf(fp, "%f", &m_origin_freq);
		fclose(fp);
	}
	else
	{
		m_origin_freq = fetchFrequency(m_wav_length, m_wav_samples, m_origin_sample_rate);
		printf("Detected frequency of %s.wav = %fHz\n", name, m_origin_freq);
		fp = fopen(filename, "w");
		fprintf(fp, "%f\n", m_origin_freq);
		fclose(fp);
	}
}

