#include "PyScoreDraft.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include <string.h>
#include <math.h>
#include <ReadWav.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


class PercussionSampler : public Percussion
{
public:
	PercussionSampler()
	{
		m_wav_length = 0;
		m_wav_samples = nullptr;
	}

	~PercussionSampler()
	{
		delete[] m_wav_samples;
	}

	bool LoadWav(const char* name)
	{
		char filename[1024];
		sprintf(filename, "PercussionSamples/%s.wav", name);

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

		return true;
	}

	virtual void GenerateBeatWave(float fNumOfSamples, BeatBuffer* beatBuf, float BufferSampleRate)
	{
		unsigned maxSample = (unsigned)((float)m_wav_length * BufferSampleRate / m_origin_sample_rate);
		beatBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
		beatBuf->Allocate();

		float mult = m_beatVolume / m_max_v;

		if (BufferSampleRate == (float)m_origin_sample_rate)
		{

			for (unsigned j = 0; j < beatBuf->m_sampleNum; j++)
			{
				float x2 = (float)j / fNumOfSamples;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				beatBuf->m_data[j] = amplitude*m_wav_samples[j] * mult;
			}
		}
		else
		{
			bool interpolation = BufferSampleRate >= (float)m_origin_sample_rate;
			for (unsigned j = 0; j < beatBuf->m_sampleNum; j++)
			{
				float x2 = (float)j / fNumOfSamples;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				float wave;
				if (interpolation)
				{
					float pos = (float)j *(float)m_origin_sample_rate / BufferSampleRate;
					int ipos1 = (int)pos;
					float frac = pos - (float)ipos1;
					int ipos2 = ipos1 + 1;
					if (ipos2 >= (int)m_wav_length) ipos2 = (int)m_wav_length - 1;

					// cubic interpolation
					int ipos0 = ipos1 - 1;
					if (ipos0 < 0) ipos0 = 0;

					int ipos3 = ipos1 + 2;
					if (ipos3 >= (int)m_wav_length) ipos3 = (int)m_wav_length - 1;

					float p0 = m_wav_samples[ipos0];
					float p1 = m_wav_samples[ipos1];
					float p2 = m_wav_samples[ipos2];
					float p3 = m_wav_samples[ipos3];

					wave = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
						(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
						(-0.5f*p0 + 0.5f*p2)*frac + p1;
				}
				else
				{
					int ipos1 = (int)ceilf(((float)j - 0.5f)*(float)m_origin_sample_rate / BufferSampleRate);
					int ipos2 = (int)floorf(((float)j + 0.5f)*(float)m_origin_sample_rate / BufferSampleRate);
					if (ipos1 < 0) ipos1 = 0;
					if (ipos2 >= (int)m_wav_length) ipos2 = (int)m_wav_length - 1;
					int count = ipos2 - ipos1 + 1;
					float sum = 0.0f;
					for (int ipos = ipos1; ipos <= ipos2; ipos++)
					{
						sum += m_wav_samples[ipos];
					}
					wave = sum / (float)count;
				}

				beatBuf->m_data[j] = amplitude*wave* mult;
			}
		}
	}
	
private:
	unsigned m_wav_length;
	float *m_wav_samples;
	float m_max_v;

	unsigned m_origin_sample_rate;

};

class PercussionSamplerInitializer : public PercussionInitializer
{
public:
	std::string m_name;
	std::string GetComment()
	{
		return std::string("\t# A percussion based on a single sample ") + m_name + ".wav\n";
	}
	virtual Percussion_deferred Init()
	{
		Percussion_deferred perc = Percussion_deferred::Instance<PercussionSampler>();
		perc.DownCast<PercussionSampler>()->LoadWav(m_name.data());
		return perc;
	}

};

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	static std::vector<PercussionSamplerInitializer> s_initializers;
#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	hFind = FindFirstFileA("PercussionSamples\\*.wav", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		char name[1024];
		memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
		name[strlen(ffd.cFileName) - 4] = 0;

		PercussionSamplerInitializer initializer;
		initializer.m_name=name;
		s_initializers.push_back(initializer);

	} while (FindNextFile(hFind, &ffd) != 0);
#else
	DIR *dir;
	struct dirent *entry;

	if (dir = opendir("PercussionSamples"))
	{
		while ((entry = readdir(dir)) != NULL)
		{
			const char* ext = entry->d_name + strlen(entry->d_name) - 4;
			if (strcmp(ext, ".wav") == 0)
			{
				char name[1024];
				memcpy(name, entry->d_name, strlen(entry->d_name) - 4);
				name[strlen(entry->d_name) - 4] = 0;

				PercussionSamplerInitializer initializer;
				initializer.m_name=name;
				s_initializers.push_back(initializer);
			}
		}
	}

#endif
	for (unsigned i = 0; i < s_initializers.size();i++)
		pyScoreDraft->RegisterPercussionClass(s_initializers[i].m_name.data(), &s_initializers[i], s_initializers[i].GetComment().data());

}
