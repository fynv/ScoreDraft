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
	std::string m_root;
	bool m_IsMultiSampler;

	InstrumentSamplerInitializer(){}
	virtual ~InstrumentSamplerInitializer(){}

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
	virtual ~InstrumentSingleSamplerInitializer(){}
	virtual Instrument_deferred Init()
	{
		if (!m_sample.m_wav_samples) m_sample.LoadWav(m_root.data(), m_name.data());

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
	virtual ~InstrumentMultiSamplerInitializer(){}

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
			sprintf(searchPath, "%s/InstrumentSamples/%s/*.wav", m_root.data(), m_name.data());

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
					wav->LoadWav(m_root.data(), name, m_name.data());
					m_SampleWavList.push_back(wav);

				} while (FindNextFile(hFind, &ffd) != 0);
			}
#else
			DIR *dir;
			struct dirent *entry;

			char dirPath[1024];
			sprintf(dirPath, "%s/InstrumentSamples/%s", m_root.data(), m_name.data());

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
							wav->LoadWav(m_root.data(), name, m_name.data());
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


PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	static std::vector<InstrumentSamplerInitializer_Deferred> s_initializers;

#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	char findStr[1024];
	sprintf(findStr, "%s/InstrumentSamples/*.wav", root);

	hFind = FindFirstFileA(findStr, &ffd);
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
		initializer->m_root = root;
		s_initializers.push_back(initializer);

	} while (FindNextFile(hFind, &ffd) != 0);


	// build multi-samplers
	sprintf(findStr, "%s/InstrumentSamples/*", root);
	hFind = FindFirstFileA(findStr, &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
		{
			InstrumentSamplerInitializer_Deferred initializer =
				InstrumentSamplerInitializer_Deferred::Instance<InstrumentMultiSamplerInitializer>();

			initializer->m_name = ffd.cFileName;
			initializer->m_root = root;
			s_initializers.push_back(initializer);
		}

	} while (FindNextFile(hFind, &ffd) != 0);

#else
	DIR *dir;
	struct dirent *entry;

	char searchPath[1024];
	sprintf(searchPath, "%s/InstrumentSamples", root);

	if (dir = opendir(searchPath))
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
					initializer->m_root = root;
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
					initializer->m_root = root;
					s_initializers.push_back(initializer);

				}
			}

		}

	}
	

#endif

	for (unsigned i = 0; i < s_initializers.size(); i++)
		pyScoreDraft->RegisterInstrumentClass(s_initializers[i]->m_name.data(), s_initializers[i], s_initializers[i]->GetComment().data());
}
