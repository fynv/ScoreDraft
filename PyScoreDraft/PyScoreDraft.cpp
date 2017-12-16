#include <Python.h>
#include <Note.h>
#include <Deferred.h>
#include "TrackBuffer.h"
#include "instruments/PureSin.h"
#include "instruments/Sawtooth.h"
#include "instruments/NaivePiano.h"
#include "instruments/BottleBlow.h"
#include "WinWavWriter.h"

typedef Deferred<NoteSequence> NoteSequence_deferred;

#include <map>
using namespace std;

static unsigned s_lastNoteSequenceId = 0;
static map<unsigned, NoteSequence_deferred> s_NoteSequenceMap;

static unsigned s_lastTrackBufferId = 0;
static map<unsigned, TrackBuffer_deferred> s_TrackBufferMap;

typedef std::vector<TrackBuffer_deferred> TrackBuferList;
static unsigned s_lastTrackBufferListId = 0;
static map<unsigned, TrackBuferList> s_TrackBufferListMap;

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

template<class Instrument>
static PyObject* t_Play(PyObject *self, PyObject *args)
{
	unsigned SeqId;
	float volume;
	unsigned tempo;
	float RefFreq;

	if (!PyArg_ParseTuple(args, "IfIf", &SeqId, &volume, &tempo, &RefFreq))
		return NULL;

	NoteSequence_deferred seq = s_NoteSequenceMap[SeqId];
	s_lastTrackBufferId++;
	TrackBuffer_deferred buffer = s_TrackBufferMap[s_lastTrackBufferId];

	Instrument inst;
	inst.PlayNotes(*buffer, *seq, tempo, RefFreq);

	float maxV = buffer->MaxValue();
	buffer->SetVolume(volume / maxV);

	return PyLong_FromUnsignedLong(s_lastTrackBufferId);
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
	TrackBuferList& list=s_TrackBufferListMap[BufferListId];
	list.push_back(buffer);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* MixTrackBufferList(PyObject *self, PyObject *args)
{
	unsigned BufferListId;
	if (!PyArg_ParseTuple(args, "I", &BufferListId))
		return NULL;

	TrackBuferList& list = s_TrackBufferListMap[BufferListId];
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
		"PureSinPlay",
		t_Play<PureSin>,
		METH_VARARGS,
		""
	},
	{
		"SawtoothPlay",
		t_Play<Sawtooth>,
		METH_VARARGS,
		""
	},
	{
		"NaivePianoPlay",
		t_Play<NaivePiano>,
		METH_VARARGS,
		""
	},
	{
		"BottleBlowPlay",
		t_Play<BottleBlow>,
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
