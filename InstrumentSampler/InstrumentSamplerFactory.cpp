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

class InstrumentSamplerFactory : public InstrumentFactory
{
public:
	InstrumentSamplerFactory()
	{
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
			m_InstList.push_back(name);
			m_IsMultiSampler.push_back(false);

		} while (FindNextFile(hFind, &ffd) != 0);


		// build multi-samplers
		hFind = FindFirstFileA("InstrumentSamples\\*", &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return;

		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
			{
				m_InstList.push_back(ffd.cFileName);
				m_IsMultiSampler.push_back(true);
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
					m_InstList.push_back(entry->d_name);
					m_IsMultiSampler.push_back(true);
				}
				else
				{
					const char* ext=entry->d_name+ strlen(entry->d_name)-4;
					if (strcmp(ext,".wav")==0)
					{
						char name[1024];
						memcpy(name, entry->d_name, strlen(entry->d_name) - 4);
						name[strlen(entry->d_name) - 4] = 0;
						m_InstList.push_back(name);
						m_IsMultiSampler.push_back(false);
					}
				}

	    	}

	    }

#endif

	}

	virtual void GetInstrumentList(std::vector<std::string>& list)
	{
		list = m_InstList;
	}

	virtual void InitiateInstrument(unsigned clsInd, Instrument_deferred& inst)
	{
		if (m_IsMultiSampler[clsInd])
		{
			inst = Instrument_deferred::Instance<InstrumentMultiSampler>();
#ifdef _WIN32
			WIN32_FIND_DATAA ffd;
			HANDLE hFind = INVALID_HANDLE_VALUE;

			char searchPath[1024];
			sprintf(searchPath, "InstrumentSamples\\%s\\*.wav", m_InstList[clsInd].data());

			hFind = FindFirstFileA(searchPath, &ffd);
			if (INVALID_HANDLE_VALUE == hFind) return;

			do
			{
				if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

				char name[1024];
				memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
				name[strlen(ffd.cFileName) - 4] = 0;
				inst.DownCast<InstrumentMultiSampler>()->LoadWav(m_InstList[clsInd].data(), name);

			} while (FindNextFile(hFind, &ffd) != 0);
#else
			DIR *dir;
			struct dirent *entry;

			char dirPath[1024];
			sprintf(dirPath, "InstrumentSamples/%s", m_InstList[clsInd].data());

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
							inst.DownCast<InstrumentMultiSampler>()->LoadWav(m_InstList[clsInd].data(), name);
						}
					}

				}

			}
#endif
		}
		else
		{
			inst = Instrument_deferred::Instance<InstrumentSingleSampler>();
			inst.DownCast<InstrumentSingleSampler>()->LoadWav(m_InstList[clsInd].data());
		}
	}

private:
	std::vector<std::string> m_InstList;
	std::vector<bool> m_IsMultiSampler;

};

PY_SCOREDRAFT_EXTENSION_INTERFACE GetFactory()
{
	static InstrumentSamplerFactory fac;
	return &fac;
}

