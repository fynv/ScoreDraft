#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include "PyScoreDraft.h"

#include <Note.h>
#include <Beat.h>
#include <Syllable.h>

#include <Instrument.h>
#include <Percussion.h>
#include <Singer.h>

#include <Deferred.h>
#include <TrackBuffer.h>
#include <instruments/PureSin.h>
#include <instruments/Square.h>
#include <instruments/Sawtooth.h>
#include <instruments/Triangle.h>
#include <instruments/NaivePiano.h>
#include <instruments/BottleBlow.h>

#include "WavIO.h"

#include <vector>
#include <utility>
#include <string>
#include <string.h>
#include <stdio.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


class StdLogger : public Logger
{
public:
	virtual void PrintLine(const char* line) const
	{
		printf("%s\n", line);
	}
};

static StdLogger s_logger;
static PyScoreDraft s_PyScoreDraft;


static PyObject* ScanExtensions(PyObject *self, PyObject *args)
{
	const char* root;
	if (!PyArg_ParseTuple(args, "s", &root))
		return PyLong_FromLong(0);

#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	char extSearchStr[1024];
	sprintf(extSearchStr, "%s/Extensions/*.dll", root);

	hFind = FindFirstFileA(extSearchStr, &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return PyLong_FromLong(0);

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		char path[1024];
		sprintf(path, "%s/Extensions/%s", root, ffd.cFileName);

		HINSTANCE hinstLib;
		hinstLib = LoadLibraryA(path);

		if (hinstLib != NULL)
		{
			typedef void (InitializeFunc)(PyScoreDraft* pyScoreDraft, const char* root);
			InitializeFunc* initFunc = (InitializeFunc*)GetProcAddress(hinstLib, "Initialize");
			if (initFunc != NULL)
			{
				printf("Loading extension: %s\n", ffd.cFileName);
				initFunc(&s_PyScoreDraft, root);
			}
		}

	} while (FindNextFile(hFind, &ffd) != 0);
#else
	DIR *dir;
	struct dirent *entry;

	char extPath[1024];
	sprintf(extPath, "%s/Extensions", root);

	if (dir = opendir(extPath))
	{
		while ((entry = readdir(dir)) != NULL)
		{
			const char* ext = entry->d_name + strlen(entry->d_name) - 3;
			if (strcmp(ext, ".so") == 0)
			{
				char path[1024];
				sprintf(path, "%s/Extensions/%s", root, entry->d_name);

				void *handle = dlopen(path, RTLD_LAZY);
				if (handle)
				{
					dlerror();
					typedef void (InitializeFunc)(PyScoreDraft* pyScoreDraft, const char* root);
					InitializeFunc* initFunc;
					*(void **)(&initFunc) = dlsym(handle, "Initialize");
					if (!dlerror())
					{
						printf("Loading extension: %s\n", entry->d_name);
						initFunc(&s_PyScoreDraft, root);
					}

				}

			}
		}
	}

#endif


	return PyLong_FromLong(0);
}

static PyObject* GenerateCode(PyObject *self, PyObject *args)
{
	std::string generatedCode = "";
	std::string summary = "";
	
	generatedCode += "# PyScoreDraft Generated Code\n\n";

	//Interfaces
	generatedCode += "# Interfaces\n\n";
	summary += "Interfaces:\n";
	
	unsigned count = s_PyScoreDraft.NumOfInterfaceExtensions();
	for (unsigned i = 0; i < count; i++)
	{
		InterfaceExtension ext = s_PyScoreDraft.GetInterfaceExtension(i);

		generatedCode +=
			std::string("def ") + ext.m_name + "(" + ext.m_input_params + "):\n"
			+ ext.m_comment
			+ "\treturn PyScoreDraft.CallExtension(" + std::to_string(i);

		if (ext.m_call_params != "") generatedCode += ",(" + ext.m_call_params + ")";

		generatedCode += ")\n\n";

		summary += std::to_string(i) + ": " + ext.m_name + "\n";
	}
	summary += "\n";

	PyObject* list = PyList_New(0);
	PyList_Append(list, _PyUnicode_FromASCII(generatedCode.data(), generatedCode.length()));
	PyList_Append(list, _PyUnicode_FromASCII(summary.data(), summary.length()));

	return list;

}

static PyObject* InitTrackBuffer(PyObject *self, PyObject *args)
{
	unsigned chn;
	if (!PyArg_ParseTuple(args, "I", &chn))
		return NULL;

	TrackBuffer_deferred buffer(44100,chn);
	unsigned id = s_PyScoreDraft.AddTrackBuffer(buffer);
	return PyLong_FromUnsignedLong((unsigned long)(id));
}

static PyObject* DelTrackBuffer(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	if (!PyArg_ParseTuple(args, "I", &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	buffer.Abondon();

	return PyLong_FromLong(0);
}

static PyObject* DelInstrument(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	if (!PyArg_ParseTuple(args, "I", &InstrumentId))
		return NULL;

	Instrument_deferred instrument = s_PyScoreDraft.GetInstrument(InstrumentId);
	instrument.Abondon();

	return PyLong_FromLong(0);
}

static PyObject* DelPercussion(PyObject *self, PyObject *args)
{
	unsigned PercussionId;
	if (!PyArg_ParseTuple(args, "I", &PercussionId))
		return NULL;

	Percussion_deferred perc = s_PyScoreDraft.GetPercussion(PercussionId);
	perc.Abondon();

	return PyLong_FromLong(0);
}

static PyObject* DelSinger(PyObject *self, PyObject *args)
{
	unsigned SingerId;
	if (!PyArg_ParseTuple(args, "I", &SingerId))
		return NULL;

	Singer_deferred singer = s_PyScoreDraft.GetSinger(SingerId);
	singer.Abondon();

	return PyLong_FromLong(0);
}

static PyObject* TrackBufferSetVolume(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	float volume;
	if (!PyArg_ParseTuple(args, "If", &BufferId, &volume))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	buffer->SetVolume(volume);

	return PyLong_FromLong(0);
}

static PyObject* TrackBufferGetVolume(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	if (!PyArg_ParseTuple(args, "I", &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	return PyFloat_FromDouble((double)buffer->Volume());
}


static PyObject* TrackBufferSetPan(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	float pan;
	if (!PyArg_ParseTuple(args, "If", &BufferId, &pan))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	buffer->SetPan(pan);

	return PyLong_FromLong(0);
}

static PyObject* TrackBufferGetPan(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	if (!PyArg_ParseTuple(args, "I", &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	return PyFloat_FromDouble((double)buffer->Pan());
}


static PyObject* TrackBufferGetNumberOfSamples(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	if (!PyArg_ParseTuple(args, "I", &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	return PyLong_FromLong((long)buffer->NumberOfSamples());
}

static PyObject* TrackBufferGetNumberOfChannels(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	if (!PyArg_ParseTuple(args, "I", &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	return PyLong_FromLong((long)buffer->NumberOfChannels());
}

static PyObject* TrackBufferGetCursor(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	if (!PyArg_ParseTuple(args, "I", &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	return PyFloat_FromDouble((double)buffer->GetCursor());
}

static PyObject* TrackBufferSetCursor(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	float cursor;
	if (!PyArg_ParseTuple(args, "If", &BufferId, &cursor))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	buffer->SetCursor(cursor);

	return PyLong_FromLong(0);
}

static PyObject* TrackBufferMoveCursor(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	float cursor_delta;
	if (!PyArg_ParseTuple(args, "If", &BufferId, &cursor_delta))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	buffer->MoveCursor(cursor_delta);

	return PyLong_FromLong(0);
}

static PyObject* InstrumentPlay(PyObject *self, PyObject *args)
{
	unsigned TrackBufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	unsigned InstrumentId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	PyObject *seq_py = PyTuple_GetItem(args, 2);
	PyObject *tempo_obj = PyTuple_GetItem(args, 3);
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(TrackBufferId);

	bool tempo_map = PyObject_TypeCheck(tempo_obj, &PyList_Type);

	unsigned tempo = (unsigned)(-1);
	TempoMap tempoMap;
	if (tempo_map)
	{
		std::pair<int, float> ctrlPnt;
		ctrlPnt.first = 0;
		ctrlPnt.second = buffer->GetCursor();
		tempoMap.push_back(ctrlPnt);

		size_t tempo_count = PyList_Size(tempo_obj);
		for (size_t i = 0; i < tempo_count; i++)
		{
			PyObject *item = PyList_GetItem(tempo_obj, i);
			ctrlPnt.first = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
			ctrlPnt.second = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 1));

			if (ctrlPnt.first == 0)
			{
				tempoMap[0].second = ctrlPnt.second;
				buffer->SetCursor(tempoMap[0].second);
			}
			else
			{
				tempoMap.push_back(ctrlPnt);
			}
		}
	}
	else
	{
		tempo = (unsigned)PyLong_AsUnsignedLong(tempo_obj);
	}


	Instrument_deferred instrument = s_PyScoreDraft.GetInstrument(InstrumentId);

	size_t piece_count = PyList_Size(seq_py);
	int beatPos = 0;
	for (size_t i = 0; i < piece_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);
		if (PyObject_TypeCheck(item, &PyTuple_Type))
		{
			PyObject* _item = PyTuple_GetItem(item, 0);
			if (PyObject_TypeCheck(_item, &PyUnicode_Type)) // singing
			{
				size_t tupleSize = PyTuple_Size(item);

				size_t j = 0;
				while (j < tupleSize)
				{
					j++; // by-pass lyric
					_item = PyTuple_GetItem(item, j);
					if (PyObject_TypeCheck(_item, &PyTuple_Type)) // singing note
					{
						for (; j<tupleSize; j++)
						{
							_item = PyTuple_GetItem(item, j);
							if (!PyObject_TypeCheck(_item, &PyTuple_Type)) break;

							Note note;
							note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(_item, 0));
							note.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(_item, 1));

							if (tempo_map)
							{
								instrument->PlayNote(*buffer, note, tempoMap, beatPos, RefFreq);
							}
							else
							{
								instrument->PlayNote(*buffer, note, tempo, RefFreq);
							}
							beatPos += note.m_duration;
						}
					}
					else if (PyObject_TypeCheck(_item, &PyLong_Type)) // singing rap
					{
						int duration = (int)PyLong_AsLong(PyTuple_GetItem(item, j));
						Note note;
						note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, j + 1));
						note.m_duration = duration;

						if (tempo_map)
						{
							instrument->PlayNote(*buffer, note, tempoMap, beatPos, RefFreq);
						}
						else
						{
							instrument->PlayNote(*buffer, note, tempo, RefFreq);
						}
						beatPos += note.m_duration;

						j++; // at freq1
						j++; // at freq2
						j++; // at next
					}
				}
			}
			else if (PyObject_TypeCheck(_item, &PyFloat_Type)) // note
			{
				Note note;
				note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 0));
				note.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(item, 1));

				if (tempo_map)
				{
					instrument->PlayNote(*buffer, note, tempoMap, beatPos, RefFreq);
				}
				else
				{
					instrument->PlayNote(*buffer, note, tempo, RefFreq);
				}
				beatPos += note.m_duration;
			}
		}
		else if (PyObject_TypeCheck(item, &PyUnicode_Type))
		{
			instrument->Tune(_PyUnicode_AsString(item));
		}
	}

	return PyLong_FromUnsignedLong(0);
}

static PyObject* InstrumentTune(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	const char* cmd;

	if (!PyArg_ParseTuple(args, "Is", &InstrumentId, &cmd))
		return NULL;

	Instrument_deferred instrument = s_PyScoreDraft.GetInstrument(InstrumentId);
	instrument->Tune(cmd);
	return PyLong_FromLong(0);
}

static PyObject* InstrumentSetNoteVolume(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	float volume;

	if (!PyArg_ParseTuple(args, "If", &InstrumentId, &volume))
		return NULL;

	Instrument_deferred instrument = s_PyScoreDraft.GetInstrument(InstrumentId);
	instrument->SetNoteVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* InstrumentSetNotePan(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	float pan;

	if (!PyArg_ParseTuple(args, "If", &InstrumentId, &pan))
		return NULL;

	Instrument_deferred instrument = s_PyScoreDraft.GetInstrument(InstrumentId);
	instrument->SetNotePan(pan);
	return PyLong_FromLong(0);
}

static PyObject* PercussionPlay(PyObject *self, PyObject *args)
{
	unsigned TrackBufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	PyObject *percId_list = PyTuple_GetItem(args, 1);
	PyObject *seq_py = PyTuple_GetItem(args, 2);
	PyObject *tempo_obj = PyTuple_GetItem(args, 3);

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(TrackBufferId);

	bool tempo_map = PyObject_TypeCheck(tempo_obj, &PyList_Type);

	unsigned tempo = (unsigned)(-1);
	TempoMap tempoMap;
	if (tempo_map)
	{
		std::pair<int, float> ctrlPnt;
		ctrlPnt.first = 0;
		ctrlPnt.second = buffer->GetCursor();
		tempoMap.push_back(ctrlPnt);

		size_t tempo_count = PyList_Size(tempo_obj);
		for (size_t i = 0; i < tempo_count; i++)
		{
			PyObject *item = PyList_GetItem(tempo_obj, i);
			ctrlPnt.first = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
			ctrlPnt.second = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 1));

			if (ctrlPnt.first == 0)
			{
				tempoMap[0].second = ctrlPnt.second;
				buffer->SetCursor(tempoMap[0].second);
			}
			else
			{
				tempoMap.push_back(ctrlPnt);
			}
		}
	}
	else
	{
		tempo = (unsigned)PyLong_AsUnsignedLong(tempo_obj);
	}

	size_t perc_count = PyList_Size(percId_list);
	Percussion_deferred *perc_List = new Percussion_deferred[perc_count];
	for (size_t i = 0; i < perc_count; i++)
	{
		unsigned long percId = PyLong_AsUnsignedLong(PyList_GetItem(percId_list, i));
		perc_List[i] = s_PyScoreDraft.GetPercussion(percId);
	}

	size_t beat_count = PyList_Size(seq_py);
	int beatPos = 0;
	for (size_t i = 0; i < beat_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);
		int percId = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));

		PyObject *operation = PyTuple_GetItem(item, 1);
		if (PyObject_TypeCheck(operation, &PyLong_Type))
		{
			int duration = (int)PyLong_AsLong(operation);

			if (tempo_map)
			{
				if (percId >= 0)
					perc_List[percId]->PlayBeat(*buffer, duration, tempoMap, beatPos);
				else if (duration >= 0)
					Percussion::PlaySilence(*buffer, duration, tempoMap, beatPos);
				else
					Percussion::PlayBackspace(*buffer, -duration, tempoMap, beatPos);
			}
			else
			{

				if (percId >= 0)
					perc_List[percId]->PlayBeat(*buffer, duration, tempo);
				else if (duration >= 0)
					Percussion::PlaySilence(*buffer, duration, tempo);
				else
					Percussion::PlayBackspace(*buffer, -duration, tempo);
			}

			beatPos += duration;
		}
		else if (PyObject_TypeCheck(operation, &PyUnicode_Type))
		{
			perc_List[percId]->Tune(_PyUnicode_AsString(operation));
		}
	}

	delete[] perc_List;

	return PyLong_FromUnsignedLong(0);
}

static PyObject* PercussionTune(PyObject *self, PyObject *args)
{
	unsigned PercussionId;
	const char* cmd;

	if (!PyArg_ParseTuple(args, "Is", &PercussionId, &cmd))
		return NULL;

	Percussion_deferred perc = s_PyScoreDraft.GetPercussion(PercussionId);
	perc->Tune(cmd);
	return PyLong_FromLong(0);
}


static PyObject* PercussionSetBeatVolume(PyObject *self, PyObject *args)
{
	unsigned PercussionId;
	float volume;

	if (!PyArg_ParseTuple(args, "If", &PercussionId, &volume))
		return NULL;

	Percussion_deferred perc = s_PyScoreDraft.GetPercussion(PercussionId);
	perc->SetBeatVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* PercussionSetBeatPan(PyObject *self, PyObject *args)
{
	unsigned PercussionId;
	float pan;

	if (!PyArg_ParseTuple(args, "If", &PercussionId, &pan))
		return NULL;

	Percussion_deferred perc = s_PyScoreDraft.GetPercussion(PercussionId);
	perc->SetBeatPan(pan);
	return PyLong_FromLong(0);
}

static PyObject* Sing(PyObject *self, PyObject *args)
{
	unsigned TrackBufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	unsigned SingerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	PyObject *seq_py = PyTuple_GetItem(args, 2);

	PyObject *tempo_obj = PyTuple_GetItem(args, 3);
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(TrackBufferId);

	bool tempo_map = PyObject_TypeCheck(tempo_obj, &PyList_Type);

	unsigned tempo = (unsigned)(-1);
	TempoMap tempoMap;
	if (tempo_map)
	{
		std::pair<int, float> ctrlPnt;
		ctrlPnt.first = 0;
		ctrlPnt.second = buffer->GetCursor();
		tempoMap.push_back(ctrlPnt);

		size_t tempo_count = PyList_Size(tempo_obj);
		for (size_t i = 0; i < tempo_count; i++)
		{
			PyObject *item = PyList_GetItem(tempo_obj, i);
			ctrlPnt.first = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
			ctrlPnt.second = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 1));

			if (ctrlPnt.first == 0)
			{
				tempoMap[0].second = ctrlPnt.second;
				buffer->SetCursor(tempoMap[0].second);
			}
			else
			{
				tempoMap.push_back(ctrlPnt);
			}
		}
	}
	else
	{
		tempo = (unsigned)PyLong_AsUnsignedLong(tempo_obj);
	}

	Singer_deferred singer = s_PyScoreDraft.GetSinger(SingerId);
	std::string lyric_charset = singer->GetLyricCharset();

	size_t piece_count = PyList_Size(seq_py);
	int beatPos = 0;
	for (size_t i = 0; i < piece_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);
		if (PyObject_TypeCheck(item, &PyTuple_Type))
		{
			PyObject *_item = PyTuple_GetItem(item, 0);
			if (PyObject_TypeCheck(_item, &PyUnicode_Type)) // singing
			{
				int totalDuration = 0;

				SyllableSequence syllables;
				size_t tupleSize = PyTuple_Size(item);

				size_t j = 0;
				while (j < tupleSize)
				{
					_item = PyTuple_GetItem(item, j);
					PyObject *byteCode = PyUnicode_AsEncodedString(_item, lyric_charset.data(), 0);
					std::string lyric = PyBytes_AS_STRING(byteCode);
					j++;

					_item = PyTuple_GetItem(item, j);
					if (PyObject_TypeCheck(_item, &PyTuple_Type)) // singing note
					{
						Syllable syllable;
						syllable.m_lyric = lyric;

						for (; j<tupleSize; j++)
						{
							_item = PyTuple_GetItem(item, j);
							if (!PyObject_TypeCheck(_item, &PyTuple_Type)) break;

							unsigned numCtrlPnt = (unsigned)(PyTuple_Size(_item)+1)/2;

							for (unsigned k = 0; k < numCtrlPnt; k++)
							{
								ControlPoint ctrlPnt;
								ctrlPnt.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(_item, k*2));
								if (k * 2 + 1 < PyTuple_Size(_item))
								{
									ctrlPnt.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(_item, k * 2 + 1));
									totalDuration += ctrlPnt.m_duration; 
								}
								else
									ctrlPnt.m_duration = 0;
								syllable.m_ctrlPnts.push_back(ctrlPnt);
							}
							ControlPoint& lastCtrlPnt = *(syllable.m_ctrlPnts.end()-1);
							if (lastCtrlPnt.m_freq_rel > 0.0f && lastCtrlPnt.m_duration>0)
							{
								ControlPoint ctrlPnt;
								ctrlPnt.m_freq_rel = lastCtrlPnt.m_freq_rel;
								ctrlPnt.m_duration = 0;
								syllable.m_ctrlPnts.push_back(ctrlPnt);
							}
						}
						syllables.push_back(syllable);
					}
					else if (PyObject_TypeCheck(_item, &PyLong_Type)) // singing rap
					{
						Syllable syllable;
						syllable.m_lyric = lyric;

						int duration;
						float freq1, freq2;

						duration = (int)PyLong_AsLong(PyTuple_GetItem(item, j));
						j++;
						freq1 = (float)PyFloat_AsDouble(PyTuple_GetItem(item, j));
						j++;
						freq2 = (float)PyFloat_AsDouble(PyTuple_GetItem(item, j));
						j++;

						if (freq1 > 0.0f && freq2 > 0.0f)
						{
							ControlPoint ctrlPnt;
							ctrlPnt.m_duration = duration;
							ctrlPnt.m_freq_rel = freq1;
							syllable.m_ctrlPnts.push_back(ctrlPnt);
							ctrlPnt.m_duration = 0;
							ctrlPnt.m_freq_rel = freq2;
							syllable.m_ctrlPnts.push_back(ctrlPnt);
						}
						else
						{
							ControlPoint ctrlPnt;
							ctrlPnt.m_duration = duration;
							ctrlPnt.m_freq_rel = -1.0f;
							syllable.m_ctrlPnts.push_back(ctrlPnt);
						}
						totalDuration += duration;

						syllables.push_back(syllable);
					}

				}
				if (syllables.size() > 0)
				{
					if (syllables.size() < 2)
					{
						if (tempo_map)
						{
							singer->SingSyllable(*buffer, syllables[0], tempoMap, beatPos, RefFreq);
						}							
						else
						{
							singer->SingSyllable(*buffer, syllables[0], tempo, RefFreq);
						}
					}
					else
					{
						if (tempo_map)
						{
							singer->SingConsecutiveSyllables(*buffer, syllables, tempoMap, beatPos, RefFreq);
						}
						else
						{
							singer->SingConsecutiveSyllables(*buffer, syllables, tempo, RefFreq);
						}
					}
				}
				beatPos += totalDuration;
			}
			else if (PyObject_TypeCheck(_item, &PyFloat_Type)) // note
			{
				Syllable syllable;
				syllable.m_lyric = "";

				ControlPoint ctrlPnt;
				ctrlPnt.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 0));
				ctrlPnt.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(item, 1));
				syllable.m_ctrlPnts.push_back(ctrlPnt);
				if (ctrlPnt.m_freq_rel>0.0f)
				{
					ctrlPnt.m_duration = 0;
					syllable.m_ctrlPnts.push_back(ctrlPnt);
				}
				if (tempo_map)
				{
					singer->SingSyllable(*buffer, syllable, tempoMap, beatPos, RefFreq);
				}
				else
				{
					singer->SingSyllable(*buffer, syllable, tempo, RefFreq);
				}
				beatPos += ctrlPnt.m_duration;
			}

		}
		else if (PyObject_TypeCheck(item, &PyUnicode_Type))
		{
			singer->Tune(_PyUnicode_AsString(item));
		}

	}

	return PyLong_FromUnsignedLong(0);
}

static PyObject* SingerTune(PyObject *self, PyObject *args)
{
	unsigned SingerId;
	const char* cmd;

	if (!PyArg_ParseTuple(args, "Is", &SingerId, &cmd))
		return NULL;

	Singer_deferred singer = s_PyScoreDraft.GetSinger(SingerId);
	singer->Tune(cmd);
	return PyLong_FromLong(0);
}

static PyObject* SingerSetDefaultLyric(PyObject *self, PyObject *args)
{
	unsigned SingerId;
	const char* lyric;

	if (!PyArg_ParseTuple(args, "Is", &SingerId, &lyric))
		return NULL;

	Singer_deferred singer = s_PyScoreDraft.GetSinger(SingerId);
	singer->SetDefaultLyric(lyric);
	return PyLong_FromLong(0);
}


static PyObject* SingerSetNoteVolume(PyObject *self, PyObject *args)
{
	unsigned SingerId;
	float volume;

	if (!PyArg_ParseTuple(args, "If", &SingerId, &volume))
		return NULL;

	Singer_deferred singer = s_PyScoreDraft.GetSinger(SingerId);
	singer->SetNoteVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* SingerSetNotePan(PyObject *self, PyObject *args)
{
	unsigned SingerId;
	float pan;

	if (!PyArg_ParseTuple(args, "If", &SingerId, &pan))
		return NULL;

	Singer_deferred singer = s_PyScoreDraft.GetSinger(SingerId);
	singer->SetNotePan(pan);
	return PyLong_FromLong(0);
}

static PyObject* MixTrackBufferList(PyObject *self, PyObject *args)
{
	unsigned TargetTrackBufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	PyObject *list = PyTuple_GetItem(args, 1);

	TrackBuffer_deferred targetBuffer = s_PyScoreDraft.GetTrackBuffer(TargetTrackBufferId);
	
	size_t bufferCount = PyList_Size(list);
	TrackBuffer_deferred* bufferList = new TrackBuffer_deferred[bufferCount];
	for (size_t i = 0; i < bufferCount; i++)
	{
		unsigned long listId = PyLong_AsUnsignedLong(PyList_GetItem(list, i));
		bufferList[i] = s_PyScoreDraft.GetTrackBuffer(listId);
	}

	targetBuffer->CombineTracks((unsigned)bufferCount, bufferList);
	delete[] bufferList;

	return PyLong_FromUnsignedLong(0);
}

static PyObject* WriteTrackBufferToWav(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	const char* fn;
	if (!PyArg_ParseTuple(args, "Is", &BufferId, &fn))
		return NULL;

	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	WriteToWav(*buffer, fn);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* ReadTrackBufferFromWav(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	const char* fn;
	if (!PyArg_ParseTuple(args, "Is", &BufferId, &fn))
		return NULL;
	TrackBuffer_deferred buffer = s_PyScoreDraft.GetTrackBuffer(BufferId);
	ReadFromWav(*buffer, fn);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* CallExtension(PyObject *self, PyObject *args)
{
	unsigned extId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	PyObject* params;
	if (PyTuple_Size(args)<2) params = PyTuple_New(0);
	else params = PyTuple_GetItem(args, 1);

	InterfaceExtension ext = s_PyScoreDraft.GetInterfaceExtension(extId);
	PyObject* ret=ext.m_func(params);
	
	return ret;
}

static PyObject* TellDuration(PyObject *self, PyObject *args)
{
	PyObject *seq_py = PyTuple_GetItem(args, 0);
	size_t piece_count = PyList_Size(seq_py);

	unsigned dure = 0;
	for (size_t i = 0; i < piece_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);
		if (PyObject_TypeCheck(item, &PyTuple_Type))
		{
			PyObject* _item = PyTuple_GetItem(item, 0);
			if (PyObject_TypeCheck(_item, &PyUnicode_Type)) // singing
			{
				size_t tupleSize = PyTuple_Size(item);
				size_t j = 0;
				while (j < tupleSize)
				{
					j++; // by-pass lyric
					_item = PyTuple_GetItem(item, j);
					if (PyObject_TypeCheck(_item, &PyTuple_Type)) // singing note
					{
						for (; j<tupleSize; j++)
						{
							_item = PyTuple_GetItem(item, j);
							if (!PyObject_TypeCheck(_item, &PyTuple_Type)) break;

							unsigned numCtrlPnt = (unsigned)PyTuple_Size(_item) / 2;
							for (unsigned k = 0; k < numCtrlPnt; k++)
							{
								dure += (int)PyLong_AsLong(PyTuple_GetItem(_item, k * 2 + 1));
							}
						}
					}
					else if (PyObject_TypeCheck(_item, &PyLong_Type)) // singing rap
					{
						dure += (int)PyLong_AsLong(PyTuple_GetItem(item, j));
						j++; // at freq1
						j++; // at freq2
						j++; // at next
					}
				}

			}
			else if (PyObject_TypeCheck(_item, &PyFloat_Type)) // note
			{
				dure += (int)PyLong_AsLong(PyTuple_GetItem(item, 1));
			}
			else if (PyObject_TypeCheck(_item, &PyLong_Type)) // beat
			{
				dure += (int)PyLong_AsLong(PyTuple_GetItem(item, 1));
			}
		}
	}

	return PyLong_FromUnsignedLong(dure);

}

enum InternalInstrument_Type
{
	PureSin_Type,
	Square_Type,
	Triangle_Type,
	Sawtooth_Type,
	NaivePiano_Type,
	BottleBlow_Type
};


static PyObject* InitializeInternalInstrument(PyObject *self, PyObject *args)
{
	InternalInstrument_Type instrument_cls = (InternalInstrument_Type)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	Instrument_deferred inst;
	switch (instrument_cls)
	{
	case PureSin_Type:
		inst = Instrument_deferred::Instance<PureSin>();
		break;
	case Square_Type:
		inst = Instrument_deferred::Instance<Square>();
		break;
	case Triangle_Type:
		inst = Instrument_deferred::Instance<Triangle>();
		break;
	case Sawtooth_Type:
		inst = Instrument_deferred::Instance<Sawtooth>();
		break;
	case NaivePiano_Type:
		inst = Instrument_deferred::Instance<NaivePiano>();
		break;
	case BottleBlow_Type:
		inst = Instrument_deferred::Instance<BottleBlow>();
		break;
	}
	unsigned id = s_PyScoreDraft.AddInstrument(inst);
	return PyLong_FromUnsignedLong(id);
}

static PyMethodDef s_PyScoreDraftMethods[] = {
	{
		"ScanExtensions",
		ScanExtensions,
		METH_VARARGS,
		""
	},
	{
		"GenerateCode",
		GenerateCode,
		METH_VARARGS,
		""
	},
	{
		"InitTrackBuffer",
		InitTrackBuffer,
		METH_VARARGS,
		""
	},
	{
		"DelTrackBuffer",
		DelTrackBuffer,
		METH_VARARGS,
		""
	},
	{
		"DelInstrument",
		DelInstrument,
		METH_VARARGS,
		""
	},
	{
		"DelPercussion",
		DelPercussion,
		METH_VARARGS,
		""
	},
	{
		"DelSinger",
		DelSinger,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferSetVolume",
		TrackBufferSetVolume,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferGetVolume",
		TrackBufferGetVolume,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferSetPan",
		TrackBufferSetPan,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferGetPan",
		TrackBufferGetPan,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferGetNumberOfSamples",
		TrackBufferGetNumberOfSamples,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferGetNumberOfChannels",
		TrackBufferGetNumberOfChannels,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferGetCursor",
		TrackBufferGetCursor,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferSetCursor",
		TrackBufferSetCursor,
		METH_VARARGS,
		""
	},
	{
		"TrackBufferMoveCursor",
		TrackBufferMoveCursor,
		METH_VARARGS,
		""
	},
	{
		"InstrumentPlay",
		InstrumentPlay,
		METH_VARARGS,
		""
	},
	{
		"InstrumentTune",
		InstrumentTune,
		METH_VARARGS,
		""
	},
	{
		"InstrumentSetNoteVolume",
		InstrumentSetNoteVolume,
		METH_VARARGS,
		""
	},
	{
		"InstrumentSetNotePan",
		InstrumentSetNotePan,
		METH_VARARGS,
		""
	},
	{
		"PercussionPlay",
		PercussionPlay,
		METH_VARARGS,
		""
	},
	{
		"PercussionTune",
		PercussionTune,
		METH_VARARGS,
		""
	},
	{
		"PercussionSetBeatVolume",
		PercussionSetBeatVolume,
		METH_VARARGS,
		""
	},
	{
		"PercussionSetBeatPan",
		PercussionSetBeatPan,
		METH_VARARGS,
		""
	},
	{
		"Sing",
		Sing,
		METH_VARARGS,
		""
	},
	{
		"SingerTune",
		SingerTune,
		METH_VARARGS,
		""
	},
	{
		"SingerSetDefaultLyric",
		SingerSetDefaultLyric,
		METH_VARARGS,
		""
	},
	{
		"SingerSetNoteVolume",
		SingerSetNoteVolume,
		METH_VARARGS,
		""
	},
	{
		"SingerSetNotePan",
		SingerSetNotePan,
		METH_VARARGS,
		""
	},
	{
		"MixTrackBufferList",
		MixTrackBufferList,
		METH_VARARGS,
		""
	},
	{
		"WriteTrackBufferToWav",
		WriteTrackBufferToWav,
		METH_VARARGS,
		""
	},
	{
		"ReadTrackBufferFromWav",
		ReadTrackBufferFromWav,
		METH_VARARGS,
		""
	},
	{
		"CallExtension",
		CallExtension,
		METH_VARARGS,
		""
	},
	{
		"TellDuration",
		TellDuration,
		METH_VARARGS,
		""
	},
	{
		"InitializeInternalInstrument",
		InitializeInternalInstrument,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};

PyScoreDraft::PyScoreDraft()
{
	m_logger = nullptr; 
	m_PyScoreDraftMethods = s_PyScoreDraftMethods;
}

static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"PyScoreDraft_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	s_PyScoreDraftMethods
}; 

PyMODINIT_FUNC PyInit_PyScoreDraft(void) {
	return PyModule_Create(&cModPyDem);
}
