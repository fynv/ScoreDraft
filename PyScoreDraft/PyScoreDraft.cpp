#include <Python.h>

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
#include <Deferred.h>
#include "TrackBuffer.h"
#include "instruments/PureSin.h"
#include "instruments/Square.h"
#include "instruments/Sawtooth.h"
#include "instruments/Triangle.h"
#include "instruments/NaivePiano.h"
#include "instruments/BottleBlow.h"
#include "WinWavWriter.h"
#include "MIDIWriter.h"

#include <Beat.h>
#include "percussions/TestPerc.h"

#include <VoicePiece.h>

#include <vector>
#include <string.h>

static std::vector<Instrument_deferred> s_InstrumentMap;
static std::vector<Percussion_deferred> s_PercussionMap;
static std::vector<Singer_deferred> s_SingerMap;

static std::vector<TrackBuffer_deferred> s_TrackBufferMap;

static std::vector<std::string> s_AllInstruments;
static std::vector<InstrumentFactory*> s_FactoriesOfInstruments;
static std::vector<unsigned> s_ClassIndicesOfInstruments;

static std::vector<std::string> s_AllPercussions;
static std::vector<InstrumentFactory*> s_FactoriesOfPercussions;
static std::vector<unsigned> s_ClassIndicesOfPercussions;

static std::vector<std::string> s_AllSingers;
static std::vector<InstrumentFactory*> s_FactoriesOfSingers;
static std::vector<unsigned> s_ClassIndicesOfSingers;

static TypicalInstrumentFactory* GetDefaultFactory()
{
	static TypicalInstrumentFactory _defaultFactory;
	static bool initialized = false;
	if (!initialized)
	{
		_defaultFactory.AddInstrument<PureSin>("PureSin");
		_defaultFactory.AddInstrument<Square>("Square");
		_defaultFactory.AddInstrument<Triangle>("Triangle");
		_defaultFactory.AddInstrument<Sawtooth>("Sawtooth");
		_defaultFactory.AddInstrument<NaivePiano>("NaivePiano");
		_defaultFactory.AddInstrument<BottleBlow>("BottleBlow");
		_defaultFactory.AddPercussion<TestPerc>("TestPerc");
	}
	return &_defaultFactory;

}

static void RegisterFactory(InstrumentFactory* factory)
{
	{
		std::vector<std::string> instList;
		factory->GetInstrumentList(instList);

		for (size_t i = 0; i < instList.size(); i++)
		{
			printf("Registering instrument, clsId=%lu, name=%s\n", s_AllInstruments.size(), instList[i].data());
			s_AllInstruments.push_back(instList[i]);
			s_FactoriesOfInstruments.push_back(factory);
			s_ClassIndicesOfInstruments.push_back((unsigned)i);
		}
	}
	{

		std::vector<std::string> percList;
		factory->GetPercussionList(percList);

		for (size_t i = 0; i < percList.size(); i++)
		{
			printf("Registering Percussion, clsId=%lu, name=%s\n", s_AllPercussions.size(), percList[i].data());
			s_AllPercussions.push_back(percList[i]);
			s_FactoriesOfPercussions.push_back(factory);
			s_ClassIndicesOfPercussions.push_back((unsigned)i);
		}
	}

	{

		std::vector<std::string> singerList;
		factory->GetSingerList(singerList);

		for (size_t i = 0; i < singerList.size(); i++)
		{
			printf("Registering Singer, clsId=%lu, name=%s\n", s_AllSingers.size(), singerList[i].data());
			s_AllSingers.push_back(singerList[i]);
			s_FactoriesOfSingers.push_back(factory);
			s_ClassIndicesOfSingers.push_back((unsigned)i);
		}
	}

}

static bool FactoriesRegistered = false;
static void RegisterFactories()
{
	TypicalInstrumentFactory* defaultFactory = GetDefaultFactory();
	RegisterFactory(defaultFactory);

#ifdef _WIN32
	WIN32_FIND_DATAA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	hFind = FindFirstFileA("Extensions\\*.dll", &ffd);
	if (INVALID_HANDLE_VALUE == hFind) return;

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

		char path[1024];
		sprintf(path, "Extensions\\%s", ffd.cFileName);

		HINSTANCE hinstLib;
		hinstLib = LoadLibraryA(path);

		if (hinstLib != NULL)
		{
			typedef InstrumentFactory*(GetFacFunc)();
			GetFacFunc* getFacFunc = (GetFacFunc*)GetProcAddress(hinstLib, "GetFactory");
			if (getFacFunc != NULL)
			{
				printf("Loading extension: %s\n", ffd.cFileName);
				InstrumentFactory* fac = getFacFunc();
				RegisterFactory(fac);
			}
		}

	} while (FindNextFile(hFind, &ffd) != 0);
#else
	DIR *dir;
    struct dirent *entry;

    if (dir = opendir("Extensions"))
    {
	    while ((entry = readdir(dir)) != NULL)
	    {
	    	const char* ext=entry->d_name+ strlen(entry->d_name)-3;
	    	if (strcmp(ext,".so")==0)
	    	{
	    		char path[1024];
	    		sprintf(path, "Extensions/%s", entry->d_name);

	    		void *handle= dlopen(path, RTLD_LAZY);
	    		if (handle)
	    		{
	    			dlerror();
	    			typedef InstrumentFactory*(GetFacFunc)();
					GetFacFunc* getFacFunc;
					*(void **)(&getFacFunc)= dlsym(handle, "GetFactory");
					if (!dlerror()) 
					{
						printf("Loading extension: %s\n", entry->d_name);
						InstrumentFactory* fac = getFacFunc();
						RegisterFactory(fac);
					}

	    		}

	    	}
	    }
	}

#endif

	FactoriesRegistered = true;
}

static PyObject* InstrumentPlay(PyObject *self, PyObject *args)
{
	unsigned InstrumentId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	PyObject *seq_py = PyTuple_GetItem(args, 1);
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
	unsigned tempo = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));
	
	Instrument_deferred instrument = s_InstrumentMap[InstrumentId];

	TrackBuffer_deferred buffer;
	s_TrackBufferMap.push_back(buffer);
	
	size_t note_count = PyList_Size(seq_py);
	for (size_t i = 0; i < note_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);
		if (PyObject_TypeCheck(item, &PyTuple_Type))
		{
			Note note;
			note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 0));
			note.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(item, 1));

			instrument->PlayNote(*buffer, note, tempo, RefFreq);
		}
		else if (PyObject_TypeCheck(item, &PyUnicode_Type))
		{
			instrument->Tune(_PyUnicode_AsString(item));
		}

	}	

	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	return PyLong_FromUnsignedLong((unsigned long)(s_TrackBufferMap.size() - 1));
}

static PyObject* InstrumentTune(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	const char* cmd;

	if (!PyArg_ParseTuple(args, "Is", &InstrumentId, &cmd))
		return NULL;

	Instrument_deferred instrument = s_InstrumentMap[InstrumentId];
	instrument->Tune(cmd);
	return PyLong_FromLong(0);
}


static PyObject* PercussionPlay(PyObject *self, PyObject *args)
{
	PyObject *percId_list = PyTuple_GetItem(args, 0);
	PyObject *seq_py = PyTuple_GetItem(args, 1);
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
	unsigned tempo = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));

	size_t perc_count = PyList_Size(percId_list);
	Percussion_deferred *perc_List = new Percussion_deferred[perc_count];
	for (size_t i = 0; i < perc_count; i++)
	{
		unsigned long percId = PyLong_AsUnsignedLong(PyList_GetItem(percId_list, i));
		perc_List[i] = s_PercussionMap[percId];
	}

	TrackBuffer_deferred buffer;
	s_TrackBufferMap.push_back(buffer);

	size_t beat_count = PyList_Size(seq_py);
	for (size_t i = 0; i < beat_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);		
		int percId = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));

		PyObject *operation = PyTuple_GetItem(item, 1);
		if (PyObject_TypeCheck(operation, &PyLong_Type))
		{
			int duration = (int)PyLong_AsLong(operation);

			if (percId >= 0)
				perc_List[percId]->PlayBeat(*buffer, duration, tempo);
			else if (duration >= 0)
				Percussion::PlaySilence(*buffer, duration, tempo);
			else
				Percussion::PlayBackspace(*buffer, -duration, tempo);
		}
		else if (PyObject_TypeCheck(operation, &PyUnicode_Type))
		{
			perc_List[percId]->Tune(_PyUnicode_AsString(operation));
		}
	}


	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	delete[] perc_List;

	return PyLong_FromUnsignedLong((unsigned long)(s_TrackBufferMap.size() - 1));
}

static PyObject* PercussionTune(PyObject *self, PyObject *args)
{
	unsigned PercussionId;
	const char* cmd;

	if (!PyArg_ParseTuple(args, "Is", &PercussionId, &cmd))
		return NULL;

	Percussion_deferred perc = s_PercussionMap[PercussionId];
	perc->Tune(cmd);
	return PyLong_FromLong(0);
}

static PyObject* Sing(PyObject *self, PyObject *args)
{
	unsigned SingerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	PyObject *seq_py = PyTuple_GetItem(args, 1);
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
	unsigned tempo = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 3));
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));

	Singer_deferred singer = s_SingerMap[SingerId];

	TrackBuffer_deferred buffer;
	s_TrackBufferMap.push_back(buffer);

	size_t piece_count = PyList_Size(seq_py);
	for (size_t i = 0; i < piece_count; i++)
	{
		PyObject *item = PyList_GetItem(seq_py, i);
		if (PyObject_TypeCheck(item, &PyTuple_Type))
		{
			size_t tupleSize = PyTuple_Size(item);
			PyObject *_item = PyTuple_GetItem(item, 0);
			if (PyObject_TypeCheck(_item, &PyUnicode_Type))
			{
				VoicePiece piece;
				piece.m_lyric = _PyUnicode_AsString(_item);
				size_t piece_count = PyList_Size(seq_py);
				for (size_t j = 0; j < tupleSize - 1; j++)
				{
					_item = PyTuple_GetItem(item, j + 1);
					Note note;
					note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(_item, 0));
					note.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(_item, 1));
					piece.m_notes.push_back(note);
				}
				singer->SingPiece(*buffer, piece, tempo, RefFreq);
			}

		}
		else if (PyObject_TypeCheck(item, &PyUnicode_Type))
		{
			singer->Tune(_PyUnicode_AsString(item));
		}
	}
	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	return PyLong_FromUnsignedLong((unsigned long)(s_TrackBufferMap.size() - 1));
}


static PyObject* SingerTune(PyObject *self, PyObject *args)
{
	unsigned SingerId;
	const char* cmd;

	if (!PyArg_ParseTuple(args, "Is", &SingerId, &cmd))
		return NULL;

	Singer_deferred singer = s_SingerMap[SingerId];
	singer->Tune(cmd);
	return PyLong_FromLong(0);
}


static PyObject* MixTrackBufferList(PyObject *self, PyObject *args)
{
	PyObject *list = PyTuple_GetItem(args, 0);
	size_t bufferCount = PyList_Size(list);
	TrackBuffer_deferred* bufferList = new TrackBuffer_deferred[bufferCount];
	for (size_t i = 0; i < bufferCount; i++)
	{
		unsigned long listId = PyLong_AsUnsignedLong(PyList_GetItem(list, i));
		bufferList[i] = s_TrackBufferMap[listId];
	}

	TrackBuffer_deferred buffer;
	s_TrackBufferMap.push_back(buffer);

	TrackBuffer::CombineTracks(*buffer, (unsigned)bufferCount, bufferList);
	float maxV = buffer->MaxValue();
	buffer->SetVolume(1.0f / maxV);

	delete[] bufferList;

	return PyLong_FromUnsignedLong((unsigned long)(s_TrackBufferMap.size() - 1));

}

static PyObject* WriteTrackBufferToWav(PyObject *self, PyObject *args)
{
	unsigned BufferId;
	const char* fn;
	if (!PyArg_ParseTuple(args, "Is", &BufferId,&fn))
		return NULL;

	TrackBuffer_deferred buffer = s_TrackBufferMap[BufferId];
	WriteToWav(*buffer, fn);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* ListInstruments(PyObject *self, PyObject *args)
{
	PyObject* list=PyList_New(0);
	size_t count = s_AllInstruments.size();
	for (size_t i = 0; i < count; i++)
	{
		PyList_Append(list, PyUnicode_FromString(s_AllInstruments[i].data()));
	}
	return list;
}

static PyObject* InitInstrument(PyObject *self, PyObject *args)
{
	unsigned glbClsId;
	if (!PyArg_ParseTuple(args, "I", &glbClsId))
		return NULL;

	InstrumentFactory* factory = s_FactoriesOfInstruments[glbClsId];
	unsigned clsId = s_ClassIndicesOfInstruments[glbClsId];
	Instrument_deferred inst;
	factory->InitiateInstrument(clsId, inst);
	s_InstrumentMap.push_back(inst);
	return PyLong_FromUnsignedLong((unsigned long)(s_InstrumentMap.size() - 1));
}

static PyObject* ListPercussions(PyObject *self, PyObject *args)
{
	PyObject* list = PyList_New(0);
	size_t count = s_AllPercussions.size();
	for (size_t i = 0; i < count; i++)
	{
		PyList_Append(list, PyUnicode_FromString(s_AllPercussions[i].data()));
	}
	return list;
}

static PyObject* InitPercussion(PyObject *self, PyObject *args)
{
	unsigned glbClsId;
	if (!PyArg_ParseTuple(args, "I", &glbClsId))
		return NULL;

	InstrumentFactory* factory = s_FactoriesOfPercussions[glbClsId];
	unsigned clsId = s_ClassIndicesOfPercussions[glbClsId];
	Percussion_deferred perc;
	factory->InitiatePercussion(clsId, perc);
	s_PercussionMap.push_back(perc);
	return PyLong_FromUnsignedLong((unsigned long)(s_PercussionMap.size() - 1));
}


static PyObject* ListSingers(PyObject *self, PyObject *args)
{
	PyObject* list = PyList_New(0);
	size_t count = s_AllSingers.size();
	for (size_t i = 0; i < count; i++)
	{
		PyList_Append(list, PyUnicode_FromString(s_AllSingers[i].data()));
	}
	return list;
}

static PyObject* InitSinger(PyObject *self, PyObject *args)
{
	unsigned glbClsId;
	if (!PyArg_ParseTuple(args, "I", &glbClsId))
		return NULL;

	InstrumentFactory* factory = s_FactoriesOfSingers[glbClsId];
	unsigned clsId = s_ClassIndicesOfSingers[glbClsId];
	Singer_deferred singer;
	factory->InitiateSinger(clsId, singer);
	s_SingerMap.push_back(singer);
	return PyLong_FromUnsignedLong((unsigned long)(s_SingerMap.size() - 1));
}

static PyObject* WriteNoteSequencesToMidi(PyObject *self, PyObject *args)
{
	PyObject *pySeqList = PyTuple_GetItem(args, 0);
	unsigned tempo = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	float refFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
	const char* fileName = _PyUnicode_AsString(PyTuple_GetItem(args, 3));

	size_t seqCount = PyList_Size(pySeqList);

	SequenceList seqList;
	for (size_t i = 0; i < seqCount; i++)
	{
		NoteSequence_deferred seq;
		PyObject *pySeq = PyList_GetItem(pySeqList, i);
		size_t note_count = PyList_Size(pySeq);
		for (size_t j = 0; j < note_count; j++)
		{
			PyObject *item = PyList_GetItem(pySeq, j);
			if (PyObject_TypeCheck(item, &PyTuple_Type))
			{
				Note note;
				note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 0));
				note.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(item, 1));
				seq->push_back(note);
			}
		}
		seqList.push_back(seq);
	}
	WriteToMidi(seqList, tempo, refFreq, fileName);


	return PyLong_FromUnsignedLong(0);
}

static PyMethodDef PyScoreDraftMethods[] = {
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
		"ListInstruments",
		ListInstruments,
		METH_VARARGS,
		""
	},
	{
		"InitInstrument",
		InitInstrument,
		METH_VARARGS,
		""
	},
	{
		"ListPercussions",
		ListPercussions,
		METH_VARARGS,
		""
	},
	{
		"InitPercussion",
		InitPercussion,
		METH_VARARGS,
		""
	},
	{
		"ListSingers",
		ListSingers,
		METH_VARARGS,
		""
	},
	{
		"InitSinger",
		InitSinger,
		METH_VARARGS,
		""
	},
	{
		"WriteNoteSequencesToMidi",
		WriteNoteSequencesToMidi,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"PyScoreDraft_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	PyScoreDraftMethods
};

PyMODINIT_FUNC PyInit_PyScoreDraft(void) {
	if (!FactoriesRegistered) RegisterFactories();
	return PyModule_Create(&cModPyDem);
}
