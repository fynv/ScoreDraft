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

class InstrumentSingleSamplerInitializer 
{
public:
	std::string m_wav_path;
	Instrument_deferred Init()
	{
		if (!m_sample.m_wav_samples) m_sample.LoadWav(m_wav_path.data());

		Instrument_deferred inst = Instrument_deferred::Instance<InstrumentSingleSampler>();
		inst.DownCast<InstrumentSingleSampler>()->SetSample(&m_sample);
		return inst;
	}

private:
	InstrumentSample m_sample;
};


class InstrumentMultiSamplerInitializer
{
public:
	std::string m_folder_path;

	static int compareSampleWav(const void* a, const void* b)
	{
		InstrumentSample_deferred wavA = *((InstrumentSample_deferred*)a);
		InstrumentSample_deferred wavB = *((InstrumentSample_deferred*)b);

		float origin_SampleFreqA = wavA->m_origin_freq / (float)wavA->m_origin_sample_rate;
		float origin_SampleFreqB = wavB->m_origin_freq / (float)wavB->m_origin_sample_rate;

		return origin_SampleFreqA > origin_SampleFreqB ? 1 : -1;
	}


	Instrument_deferred Init()
	{
		if (m_SampleWavList.size() < 1)
		{

#ifdef _WIN32
			WIN32_FIND_DATAA ffd;
			HANDLE hFind = INVALID_HANDLE_VALUE;

			char searchPath[1024];
			sprintf(searchPath, "%s/*.wav", m_folder_path.data());

			hFind = FindFirstFileA(searchPath, &ffd);
			if (INVALID_HANDLE_VALUE != hFind)
			{
				do
				{
					if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
					char wav_path[1024];
					sprintf(wav_path, "%s/%s", m_folder_path.data(), ffd.cFileName);
					InstrumentSample_deferred wav;
					wav->LoadWav(wav_path);
					m_SampleWavList.push_back(wav);

				} while (FindNextFile(hFind, &ffd) != 0);
			}
#else
			DIR *dir;
			struct dirent *entry;

			if (dir = opendir(m_folder_path.data()))
			{
				while ((entry = readdir(dir)) != NULL)
				{
					if (entry->d_type != DT_DIR)
					{
						const char* ext = entry->d_name + strlen(entry->d_name) - 4;
						if (strcmp(ext, ".wav") == 0)
						{
							char wav_path[1024];
							sprintf(wav_path, "%s/%s", m_folder_path.data(), entry->d_name);
							InstrumentSample_deferred wav;
							wav->LoadWav(wav_path);
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

#include <map>
std::map<std::string, InstrumentSingleSamplerInitializer> s_initializers_single;
std::map<std::string, InstrumentMultiSamplerInitializer> s_initializers_multi;

InstrumentSingleSamplerInitializer* GetInitializer_Single(std::string path)
{
	if (s_initializers_single.find(path) == s_initializers_single.end())
	{
		InstrumentSingleSamplerInitializer initializer;
		initializer.m_wav_path = path;
		s_initializers_single[path] = initializer;
	}
	return &s_initializers_single[path];
}

InstrumentMultiSamplerInitializer* GetInitializer_Multi(std::string path)
{
	if (s_initializers_multi.find(path) == s_initializers_multi.end())
	{
		InstrumentMultiSamplerInitializer initializer;
		initializer.m_folder_path = path;
		s_initializers_multi[path] = initializer;
	}
	return &s_initializers_multi[path];
}

static PyScoreDraft* s_PyScoreDraft;

PyObject * InitializeInstrumentSingleSampler(PyObject *args)
{
	std::string path = _PyUnicode_AsString(args);
	InstrumentSingleSamplerInitializer* initializer = GetInitializer_Single(path);
	Instrument_deferred inst = initializer->Init();
	unsigned id = s_PyScoreDraft->AddInstrument(inst);
	return PyLong_FromUnsignedLong(id);
}

PyObject * InitializeInstrumentMultiSampler(PyObject *args)
{
	std::string path = _PyUnicode_AsString(args);
	InstrumentMultiSamplerInitializer* initializer = GetInitializer_Multi(path);
	Instrument_deferred inst = initializer->Init();
	unsigned id = s_PyScoreDraft->AddInstrument(inst);
	return PyLong_FromUnsignedLong(id);
}

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	s_PyScoreDraft = pyScoreDraft;
	pyScoreDraft->RegisterInterfaceExtension("InitializeInstrumentSingleSampler", InitializeInstrumentSingleSampler,
		"wavPath", "wavPath",
		"\t'''\n"
		"\tInitialize a instrument sampler using a single .wav file.\n"
		"\twavPath -- path to the .wav file.\n"
		"\t'''\n");
	pyScoreDraft->RegisterInterfaceExtension("InitializeInstrumentMultiSampler", InitializeInstrumentMultiSampler,
		"folderPath", "folderPath",
		"\t'''\n"
		"\tInitialize a instrument sampler using multiple .wav files.\n"
		"\folderPath -- path containining the .wav files\n"
		"\t'''\n");

}
