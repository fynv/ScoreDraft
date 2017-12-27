#include <Python.h>
#include <Windows.h>
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

#include <Beat.h>
#include "percussions/TestPerc.h"

#include <vector>

static std::vector<Instrument_deferred> s_InstrumentMap;
static std::vector<Percussion_deferred> s_PercussionMap;
static std::vector<TrackBuffer_deferred> s_TrackBufferMap;

static std::vector<std::string> s_AllInstruments;
static std::vector<InstrumentFactory*> s_FactoriesOfInstruments;
static std::vector<unsigned> s_ClassIndicesOfInstruments;

static std::vector<std::string> s_AllPercussions;
static std::vector<InstrumentFactory*> s_FactoriesOfPercussions;
static std::vector<unsigned> s_ClassIndicesOfPercussions;

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
			s_AllPercussions.push_back(percList[i]);
			s_FactoriesOfPercussions.push_back(factory);
			s_ClassIndicesOfPercussions.push_back((unsigned)i);
		}
	}

}

static bool FactoriesRegistered = false;
static void RegisterFactories()
{
	TypicalInstrumentFactory* defaultFactory = GetDefaultFactory();
	RegisterFactory(defaultFactory);

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

		typedef InstrumentFactory*(GetFacFunc)();
		GetFacFunc* getFacFunc;

		if (hinstLib != NULL)
		{
			getFacFunc = (GetFacFunc*)GetProcAddress(hinstLib, "GetFactory");
			if (getFacFunc != NULL)
			{
				InstrumentFactory* fac = getFacFunc();
				RegisterFactory(fac);
			}
		}

	} while (FindNextFile(hFind, &ffd) != 0);

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
	NoteSequence seq;
	size_t note_count = PyList_Size(seq_py);
	for (size_t i = 0; i < note_count; i++)
	{
		PyObject *note_py = PyList_GetItem(seq_py, i);
		Note note;
		note.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(note_py, 0));
		note.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(note_py, 1));
		seq.push_back(note);
	}

	TrackBuffer_deferred buffer;
	s_TrackBufferMap.push_back(buffer);

	instrument->PlayNotes(*buffer, seq, tempo, RefFreq);

	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	return PyLong_FromUnsignedLong((unsigned long)(s_TrackBufferMap.size() - 1));
}

static PyObject* InstrumentTune(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	const char* nob;
	float value;

	if (!PyArg_ParseTuple(args, "Isf", &InstrumentId, &nob, &value))
		return NULL;

	Instrument_deferred instrument = s_InstrumentMap[InstrumentId];
	instrument->Tune(nob, value);
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

	BeatSequence seq;
	size_t beat_count = PyList_Size(seq_py);
	for (size_t i = 0; i < beat_count; i++)
	{
		PyObject *beat_py = PyList_GetItem(seq_py, i);
		Beat beat;
		beat.m_PercId = (int)PyLong_AsLong(PyTuple_GetItem(beat_py, 0));
		beat.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(beat_py, 1));
		seq.push_back(beat);
	}

	TrackBuffer_deferred buffer;
	s_TrackBufferMap.push_back(buffer);
	Percussion::PlayBeats(*buffer, perc_List, seq, tempo);

	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	delete[] perc_List;

	return PyLong_FromUnsignedLong((unsigned long)(s_TrackBufferMap.size() - 1));
}

static PyObject* PercussionTune(PyObject *self, PyObject *args)
{
	unsigned PercussionId;
	const char* nob;
	float value;

	if (!PyArg_ParseTuple(args, "Isf", &PercussionId, &nob, &value))
		return NULL;

	Percussion_deferred perc = s_PercussionMap[PercussionId];
	perc->Tune(nob, value);
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
