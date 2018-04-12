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

class PercussionSample
{
public:
	unsigned m_wav_length;
	unsigned m_chn;
	float *m_wav_samples;
	float m_max_v;

	unsigned m_origin_sample_rate;

	PercussionSample()
	{
		m_wav_length = 0;
		m_wav_samples = nullptr;
	}

	~PercussionSample()
	{
		delete[] m_wav_samples;
	}

	bool LoadWav(const char* root, const char* name)
	{
		char filename[1024];
		sprintf(filename, "%s/PercussionSamples/%s.wav", root, name);

		delete[] m_wav_samples;
		m_wav_length = 0;
		m_chn = 1;
		m_wav_samples = nullptr;

		ReadWav reader;
		reader.OpenFile(filename);
		if (!reader.ReadHeader(m_origin_sample_rate, m_wav_length, m_chn)) return false;

		m_wav_samples = new float[m_wav_length*m_chn];
		if (!reader.ReadSamples(m_wav_samples, m_wav_length, m_max_v))
		{
			delete[] m_wav_samples;
			return false;
		}

		return true;
	}
};


class PercussionSampler : public Percussion
{
public:
	PercussionSampler()
	{
		m_sample = nullptr;
	}

	virtual ~PercussionSampler()
	{
	}

	void SetSample(PercussionSample* sample)
	{
		m_sample = sample;
	}

	virtual void GenerateBeatWave(float fNumOfSamples, NoteBuffer* beatBuf)
	{
		if (!m_sample) return;

		unsigned maxSample = (unsigned)((float)m_sample->m_wav_length * beatBuf->m_sampleRate / m_sample->m_origin_sample_rate);
		beatBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
		unsigned chn= beatBuf->m_channelNum = m_sample->m_chn;
		beatBuf->Allocate();

		float mult = 1.0f / m_sample->m_max_v;

		if (beatBuf->m_sampleRate == (float)m_sample->m_origin_sample_rate)
		{

			for (unsigned j = 0; j < beatBuf->m_sampleNum; j++)
			{
				float x2 = (float)j / fNumOfSamples;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				for (unsigned c = 0; c < chn; c++)
				{
					beatBuf->m_data[j * chn + c] = amplitude*m_sample->m_wav_samples[j * chn + c] * mult;
				}
			}
		}
		else
		{
			bool interpolation = beatBuf->m_sampleRate >= (float)m_sample->m_origin_sample_rate;
			for (unsigned j = 0; j < beatBuf->m_sampleNum; j++)
			{
				float x2 = (float)j / fNumOfSamples;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				float wave[2];
				if (interpolation)
				{
					float pos = (float)j *(float)m_sample->m_origin_sample_rate / beatBuf->m_sampleRate;
					int ipos1 = (int)pos;
					float frac = pos - (float)ipos1;
					int ipos2 = ipos1 + 1;
					if (ipos2 >= (int)m_sample->m_wav_length) ipos2 = (int)m_sample->m_wav_length - 1;

					// cubic interpolation
					int ipos0 = ipos1 - 1;
					if (ipos0 < 0) ipos0 = 0;

					int ipos3 = ipos1 + 2;
					if (ipos3 >= (int)m_sample->m_wav_length) ipos3 = (int)m_sample->m_wav_length - 1;

					for (unsigned c = 0; c < chn; c++)
					{
						float p0 = m_sample->m_wav_samples[ipos0*chn +c];
						float p1 = m_sample->m_wav_samples[ipos1*chn + c];
						float p2 = m_sample->m_wav_samples[ipos2*chn + c];
						float p3 = m_sample->m_wav_samples[ipos3*chn + c];

						wave[c] = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
							(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
							(-0.5f*p0 + 0.5f*p2)*frac + p1;

					}
					
				}
				else
				{
					int ipos1 = (int)ceilf(((float)j - 0.5f)*(float)m_sample->m_origin_sample_rate / beatBuf->m_sampleRate);
					int ipos2 = (int)floorf(((float)j + 0.5f)*(float)m_sample->m_origin_sample_rate / beatBuf->m_sampleRate);
					if (ipos1 < 0) ipos1 = 0;
					if (ipos2 >= (int)m_sample->m_wav_length) ipos2 = (int)m_sample->m_wav_length - 1;
					int count = ipos2 - ipos1 + 1;

					for (unsigned c = 0; c < chn; c++)
					{
						float sum = 0.0f;
						for (int ipos = ipos1; ipos <= ipos2; ipos++)
						{
							sum += m_sample->m_wav_samples[ipos*chn+c];
						}
						wave[c] = sum / (float)count;
					}
				}
				if (beatBuf->m_channelNum == 1)
				{
					beatBuf->m_data[j] = amplitude*wave[0] * mult;
				}
				else if (beatBuf->m_channelNum == 2)
				{
					beatBuf->m_data[j * 2] = amplitude*wave[0] * mult;
					beatBuf->m_data[j * 2 + 1] = amplitude*wave[1] * mult;
				}
			}
		}
	}
	
private:
	PercussionSample* m_sample;

};

class PercussionSamplerInitializer : public PercussionInitializer
{
public:
	std::string m_name;
	std::string m_root;
	std::string GetComment()
	{
		return std::string("\t# A percussion based on a single sample ") + m_name + ".wav\n";
	}
	virtual Percussion_deferred Init()
	{
		if (!m_sample.m_wav_samples) m_sample.LoadWav(m_root.data(), m_name.data());

		Percussion_deferred perc = Percussion_deferred::Instance<PercussionSampler>();
		perc.DownCast<PercussionSampler>()->SetSample(&m_sample);
		return perc;
	}
private:
	PercussionSample m_sample;
};

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	static std::vector<PercussionSamplerInitializer> s_initializers;
#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	char findStr[1024];
	sprintf(findStr, "%s/PercussionSamples/*.wav", root);

	hFind = FindFirstFileA(findStr, &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		char name[1024];
		memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
		name[strlen(ffd.cFileName) - 4] = 0;

		PercussionSamplerInitializer initializer;
		initializer.m_name = name;
		initializer.m_root = root;
		s_initializers.push_back(initializer);

	} while (FindNextFile(hFind, &ffd) != 0);
#else
	DIR *dir;
	struct dirent *entry;
	char searchPath[1024];
	sprintf(searchPath, "%s/PercussionSamples", root);

	if (dir = opendir(searchPath))
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
				initializer.m_root = root;
				s_initializers.push_back(initializer);
			}
		}
	}

#endif
	for (unsigned i = 0; i < s_initializers.size();i++)
		pyScoreDraft->RegisterPercussionClass(s_initializers[i].m_name.data(), &s_initializers[i], s_initializers[i].GetComment().data());

}
