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

#include "InstrumentSingleSampler.h"
#include "InstrumentMultiSampler.h"

class InstrumentSamplerInitializer : public InstrumentInitializer
{
public:
	std::string m_name;
	bool m_IsMultiSampler;
	virtual Instrument_deferred Init()
	{
		if (m_IsMultiSampler)
		{
			Instrument_deferred inst = Instrument_deferred::Instance<InstrumentMultiSampler>();
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
					inst.DownCast<InstrumentMultiSampler>()->LoadWav(m_name.data(), name);

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
							inst.DownCast<InstrumentMultiSampler>()->LoadWav(m_name.data(), name);
						}
					}

				}

			}
#endif
			return inst;
		}
		else
		{
			Instrument_deferred inst = Instrument_deferred::Instance<InstrumentSingleSampler>();
			inst.DownCast<InstrumentSingleSampler>()->LoadWav(m_name.data());
			return inst;
		}
	}
};

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	static std::vector<InstrumentSamplerInitializer> s_initializers;

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

		InstrumentSamplerInitializer initializer;
		initializer.m_name = name;
		initializer.m_IsMultiSampler = false;
		s_initializers.push_back(initializer);

	} while (FindNextFile(hFind, &ffd) != 0);


	// build multi-samplers
	hFind = FindFirstFileA("InstrumentSamples\\*", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
		{
			InstrumentSamplerInitializer initializer;
			initializer.m_name = ffd.cFileName;
			initializer.m_IsMultiSampler = true;
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
					InstrumentSamplerInitializer initializer;
					initializer.m_name = entry->d_name;
					initializer.m_IsMultiSampler = true;
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

					InstrumentSamplerInitializer initializer;
					initializer.m_name = name;
					initializer.m_IsMultiSampler = false;
					s_initializers.push_back(initializer);

				}
			}

		}

	}

#endif

	for (unsigned i = 0; i < s_initializers.size(); i++)
		pyScoreDraft->RegisterInstrumentClass(s_initializers[i].m_name.data(), &s_initializers[i]);
}
