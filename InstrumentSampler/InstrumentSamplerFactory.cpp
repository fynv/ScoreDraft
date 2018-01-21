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

#include "InstrumentSample.h"
#include "InstrumentSingleSampler.h"
#include "InstrumentMultiSampler.h"

#include <Deferred.h>

class InstrumentSamplerInitializer : public InstrumentInitializer
{
public:
	std::string m_name;
	bool m_IsMultiSampler;

	std::string GetComment()
	{
		if (!m_IsMultiSampler)
			return std::string("\t# An instrument based on a single sample ") + m_name + ".wav\n";
		else
			return std::string("\t# An instrument based on samples in directory ") + m_name + "\n";
	}
};

typedef Deferred<InstrumentSamplerInitializer> InstrumentSamplerInitializer_Deferred;

class InstrumentSingleSamplerInitializer : public InstrumentSamplerInitializer
{
public:
	InstrumentSingleSamplerInitializer()
	{
		m_IsMultiSampler = false;
	}
	virtual Instrument_deferred Init()
	{
		if (!m_sample.m_wav_samples) m_sample.LoadWav(m_name.data());

		Instrument_deferred inst = Instrument_deferred::Instance<InstrumentSingleSampler>();
		inst.DownCast<InstrumentSingleSampler>()->SetSample(&m_sample);
		return inst;
	}

private:
	InstrumentSample m_sample;
};


class InstrumentMultiSamplerInitializer : public InstrumentSamplerInitializer
{
public:
	InstrumentMultiSamplerInitializer()
	{
		m_IsMultiSampler = true;
	}


	static int compareSampleWav(const void* a, const void* b)
	{
		InstrumentSample_deferred wavA = *((InstrumentSample_deferred*)a);
		InstrumentSample_deferred wavB = *((InstrumentSample_deferred*)b);

		float origin_SampleFreqA = wavA->m_origin_freq / (float)wavA->m_origin_sample_rate;
		float origin_SampleFreqB = wavB->m_origin_freq / (float)wavB->m_origin_sample_rate;

		return origin_SampleFreqA > origin_SampleFreqB ? 1 : -1;
	}


	virtual Instrument_deferred Init()
	{
		if (m_SampleWavList.size() < 1)
		{

#ifdef _WIN32
			WIN32_FIND_DATAA ffd;
			HANDLE hFind = INVALID_HANDLE_VALUE;

			char searchPath[1024];
			sprintf(searchPath, "InstrumentSamples\\%s\\*.wav", m_name.data());

			hFind = FindFirstFileA(searchPath, &ffd);
			if (INVALID_HANDLE_VALUE != hFind)
			{
				do
				{
					if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
					char name[1024];
					memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
					name[strlen(ffd.cFileName) - 4] = 0;

					InstrumentSample_deferred wav;
					wav->LoadWav(name, m_name.data());
					m_SampleWavList.push_back(wav);

				} while (FindNextFile(hFind, &ffd) != 0);
			}
#else
			DIR *dir;
			struct dirent *entry;

			char dirPath[1024];
			sprintf(dirPath, "InstrumentSamples/%s", m_name.data());

			if (dir = opendir(dirPath))
			{
				while ((entry = readdir(dir)) != NULL)
				{
					if (entry->d_type != DT_DIR)
					{
						const char* ext = entry->d_name + strlen(entry->d_name) - 4;
						if (strcmp(ext, ".wav") == 0)
						{
							char name[1024];
							memcpy(name, entry->d_name, strlen(entry->d_name) - 4);
							name[strlen(entry->d_name) - 4] = 0;

							InstrumentSample_deferred wav;
							wav->LoadWav(name, m_name.data());
							m_SampleWavList.push_back(wav);
						}
					}

				}

			}
#endif
			std::qsort(m_SampleWavList.data(), m_SampleWavList.size(), sizeof(InstrumentSample_deferred), compareSampleWav);
		}
		Instrument_deferred inst = Instrument_deferred::Instance<InstrumentMultiSampler>();
		inst.DownCast<InstrumentMultiSampler>()->SetSampleList(&m_SampleWavList);
		return inst;

	}

private:
	std::vector<InstrumentSample_deferred> m_SampleWavList;
};


PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	static std::vector<InstrumentSamplerInitializer_Deferred> s_initializers;

#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	hFind = FindFirstFileA("InstrumentSamples\\*.wav", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		char name[1024];
		memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
		name[strlen(ffd.cFileName) - 4] = 0;

		InstrumentSamplerInitializer_Deferred initializer=
			InstrumentSamplerInitializer_Deferred::Instance<InstrumentSingleSamplerInitializer>();
	
		initializer->m_name = name;
		s_initializers.push_back(initializer);

	} while (FindNextFile(hFind, &ffd) != 0);


	// build multi-samplers
	hFind = FindFirstFileA("InstrumentSamples\\*", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
		{
			InstrumentSamplerInitializer_Deferred initializer =
				InstrumentSamplerInitializer_Deferred::Instance<InstrumentMultiSamplerInitializer>();

			initializer->m_name = ffd.cFileName;
			s_initializers.push_back(initializer);
		}

	} while (FindNextFile(hFind, &ffd) != 0);

#else
	DIR *dir;
	struct dirent *entry;

	if (dir = opendir("InstrumentSamples"))
	{
		while ((entry = readdir(dir)) != NULL)
		{
			if (entry->d_type == DT_DIR)
			{
				if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
				{
					InstrumentSamplerInitializer_Deferred initializer=
						InstrumentSamplerInitializer_Deferred::Instance<InstrumentMultiSamplerInitializer>();
					initializer->m_name = entry->d_name;
					s_initializers.push_back(initializer);
				}
			}
			else
			{
				const char* ext = entry->d_name + strlen(entry->d_name) - 4;
				if (strcmp(ext, ".wav") == 0)
				{
					char name[1024];
					memcpy(name, entry->d_name, strlen(entry->d_name) - 4);
					name[strlen(entry->d_name) - 4] = 0;

					InstrumentSamplerInitializer_Deferred initializer =
						InstrumentSamplerInitializer_Deferred::Instance<InstrumentSingleSamplerInitializer>();
					initializer->m_name = name;
					s_initializers.push_back(initializer);

				}
			}

		}

	}

#endif

	for (unsigned i = 0; i < s_initializers.size(); i++)
		pyScoreDraft->RegisterInstrumentClass(s_initializers[i]->m_name.data(), s_initializers[i], s_initializers[i]->GetComment().data());
}
