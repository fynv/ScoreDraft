#include <Python.h>
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

typedef Deferred<NoteSequence> NoteSequence_deferred;
typedef Deferred<BeatSequence> BeatSequence_deferred;

#include <map>
using namespace std;

static unsigned s_lastNoteSequenceId = 0;
static map<unsigned, NoteSequence_deferred> s_NoteSequenceMap;

static unsigned s_lastBeatSequenceId = 0;
static map<unsigned, BeatSequence_deferred> s_BeatSequenceMap;

static unsigned s_lastTrackBufferId = 0;
static map<unsigned, TrackBuffer_deferred> s_TrackBufferMap;

typedef std::vector<TrackBuffer_deferred> TrackBufferList;
static unsigned s_lastTrackBufferListId = 0;
static map<unsigned, TrackBufferList> s_TrackBufferListMap;

typedef Deferred<Instrument> Instrument_deferred;
static unsigned s_lastInstrumentId = 0;
static map<unsigned, Instrument_deferred> s_InstrumentMap;

typedef Deferred<Percussion> Percussion_deferred;
static unsigned s_lastPercussionId = 0;
static map<unsigned, Percussion_deferred> s_PercussionMap;

typedef std::vector<Percussion_deferred> PercussionList;
static unsigned s_lastPercussionListId = 0;
static map<unsigned, PercussionList> s_PercussionListMap;

static PyObject* InitNoteSequence(PyObject *self, PyObject *args)
{
	s_lastNoteSequenceId++;
	return PyLong_FromUnsignedLong(s_lastNoteSequenceId);
}

static PyObject* AddNoteToSequence(PyObject *self, PyObject *args)
{
	unsigned SeqId;
	Note note;

	if (!PyArg_ParseTuple(args, "Ifi", &SeqId, &note.m_freq_rel, &note.m_duration))
		return NULL;

	NoteSequence_deferred seq = s_NoteSequenceMap[SeqId];
	seq->push_back(note);
	return PyLong_FromLong(0);
}

static PyObject* InitBeatSequence(PyObject *self, PyObject *args)
{
	s_lastBeatSequenceId++;
	return PyLong_FromUnsignedLong(s_lastBeatSequenceId);
}


static PyObject* AddBeatToSequence(PyObject *self, PyObject *args)
{
	unsigned SeqId;
	Beat beat;

	if (!PyArg_ParseTuple(args, "Iii", &SeqId, &beat.m_PercId, &beat.m_duration))
		return NULL;

	BeatSequence_deferred seq = s_BeatSequenceMap[SeqId];
	seq->push_back(beat);
	return PyLong_FromLong(0);
}

template<class T_Instrument>
static PyObject* t_InitInstrument(PyObject *self, PyObject *args)
{
	s_lastInstrumentId++;
	s_InstrumentMap[s_lastInstrumentId] = Instrument_deferred::Instance<T_Instrument>();
	return PyLong_FromUnsignedLong(s_lastInstrumentId);
}

static PyObject* InstrumentPlay(PyObject *self, PyObject *args)
{
	unsigned InstrumentId;
	unsigned SeqId;
	float volume;
	unsigned tempo;
	float RefFreq;

	if (!PyArg_ParseTuple(args, "IIfIf", &InstrumentId, &SeqId, &volume, &tempo, &RefFreq))
		return NULL;

	Instrument_deferred instrument = s_InstrumentMap[InstrumentId];

	NoteSequence_deferred seq = s_NoteSequenceMap[SeqId];
	s_lastTrackBufferId++;
	TrackBuffer_deferred buffer = s_TrackBufferMap[s_lastTrackBufferId];

	instrument->PlayNotes(*buffer, *seq, tempo, RefFreq);

	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	return PyLong_FromUnsignedLong(s_lastTrackBufferId);
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


template<class T_Percussion>
static PyObject* t_InitPercussion(PyObject *self, PyObject *args)
{
	s_lastPercussionId++;
	s_PercussionMap[s_lastPercussionId] = Percussion_deferred::Instance<T_Percussion>();
	return PyLong_FromUnsignedLong(s_lastPercussionId);
}


static PyObject* InitPercussionList(PyObject *self, PyObject *args)
{
	s_lastPercussionListId++;
	return PyLong_FromUnsignedLong(s_lastPercussionListId);
}

static PyObject* AddPercussionToList(PyObject *self, PyObject *args)
{
	unsigned PercussionListId;
	unsigned PercussionId;

	if (!PyArg_ParseTuple(args, "II", &PercussionListId, &PercussionId))
		return NULL;

	Percussion_deferred perc = s_PercussionMap[PercussionId];
	PercussionList& list = s_PercussionListMap[PercussionListId];
	list.push_back(perc);

	return PyLong_FromUnsignedLong(0);
}


static PyObject* PercussionPlay(PyObject *self, PyObject *args)
{
	unsigned PercussionListId;
	unsigned SeqId;
	float volume;
	unsigned tempo;

	if (!PyArg_ParseTuple(args, "IIfI", &PercussionListId, &SeqId, &volume, &tempo))
		return NULL;

	PercussionList& percList = s_PercussionListMap[PercussionListId];

	BeatSequence_deferred seq = s_BeatSequenceMap[SeqId];
	s_lastTrackBufferId++;
	TrackBuffer_deferred buffer = s_TrackBufferMap[s_lastTrackBufferId];

	Percussion::PlayBeats(*buffer, percList.data(), *seq, tempo);

	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	return PyLong_FromUnsignedLong(s_lastTrackBufferId);
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

static PyObject* InitTrackBufferList(PyObject *self, PyObject *args)
{
	s_lastTrackBufferListId++;
	return PyLong_FromUnsignedLong(s_lastTrackBufferListId);
}

static PyObject* AddTrackBufferToList(PyObject *self, PyObject *args)
{
	unsigned BufferListId;
	unsigned BufferId;

	if (!PyArg_ParseTuple(args, "II", &BufferListId, &BufferId))
		return NULL;

	TrackBuffer_deferred buffer = s_TrackBufferMap[BufferId];
	TrackBufferList& list=s_TrackBufferListMap[BufferListId];
	list.push_back(buffer);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* MixTrackBufferList(PyObject *self, PyObject *args)
{
	unsigned BufferListId;
	if (!PyArg_ParseTuple(args, "I", &BufferListId))
		return NULL;

	TrackBufferList& list = s_TrackBufferListMap[BufferListId];
	s_lastTrackBufferId++;
	TrackBuffer_deferred buffer = s_TrackBufferMap[s_lastTrackBufferId];

	TrackBuffer::CombineTracks(*buffer, (unsigned)list.size(), list.data());
	float maxV = buffer->MaxValue();
	buffer->SetVolume(1.0f / maxV);

	return PyLong_FromUnsignedLong(s_lastTrackBufferId);
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


static PyMethodDef PyScoreDraftMethods[] = {
	{
		"InitNoteSequence",
		InitNoteSequence,
		METH_VARARGS,
		""
	},
	{
		"AddNoteToSequence",
		AddNoteToSequence,
		METH_VARARGS,
		""
	},
	{
		"InitBeatSequence",
		InitBeatSequence,
		METH_VARARGS,
		""
	},
	{
		"AddBeatToSequence",
		AddBeatToSequence,
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
		"InitPercussionList",
		InitPercussionList,
		METH_VARARGS,
		""
	},
	{
		"AddPercussionToList",
		AddPercussionToList,
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
		"InitTrackBufferList",
		InitTrackBufferList,
		METH_VARARGS,
		""
	},
	{
		"AddTrackBufferToList",
		AddTrackBufferToList,
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
		"InitPureSin",
		t_InitInstrument<PureSin>,
		METH_VARARGS,
		""
	},
	{
		"InitSquare",
		t_InitInstrument<Square>,
		METH_VARARGS,
		""
	},
	{
		"InitTriangle",
		t_InitInstrument<Triangle>,
		METH_VARARGS,
		""
	},
	{
		"InitSawtooth",
		t_InitInstrument<Sawtooth>,
		METH_VARARGS,
		""
	},
	{
		"InitNaivePiano",
		t_InitInstrument<NaivePiano>,
		METH_VARARGS,
		""
	},
	{
		"InitBottleBlow",
		t_InitInstrument<BottleBlow>,
		METH_VARARGS,
		""
	},
	{
		"InitTestPerc",
		t_InitPercussion<TestPerc>,
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
	return PyModule_Create(&cModPyDem);
}
