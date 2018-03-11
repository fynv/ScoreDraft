#include "PyScoreDraft.h"
#include "Deferred.h"
#include <string.h>
#include "Meteor.h"
#include "MainWidget.h"
#include <qapplication.h>

static PyScoreDraft* s_PyScoreDraft;

void Visualizer::ProcessNoteSeq(unsigned instrumentId, float startPosition, PyObject *seq_py, unsigned tempo, float RefFreq)
{
	float pos = startPosition;

	int pitchShift = (int)(logf(RefFreq / 261.626f)*12.0f / logf(2.0f) +0.5f);

	size_t piece_count = PyList_Size(seq_py);
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

							float freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(_item, 0));
							int duration = (int)PyLong_AsLong(PyTuple_GetItem(_item, 1));
							float fduration = (float)(duration * 60) / (float)(tempo * 48);
							if (freq_rel >0.0f)
							{

								VisNote note;
								note.instrumentId = instrumentId;
								note.pitch = (int)floorf(logf(freq_rel)*12.0f / logf(2.0f) + 0.5f) + pitchShift;
								note.start = pos;
								note.end = pos + fduration;
								m_notes.push_back(note);
							}
							pos += fduration;

						}
					}
					else if (PyObject_TypeCheck(_item, &PyLong_Type)) // singing rap
					{
						float freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, j + 1));
						int duration = (int)PyLong_AsLong(PyTuple_GetItem(item, j));
						float fduration = (float)(duration * 60) / (float)(tempo * 48);

						if (freq_rel >0.0f)
						{
							VisNote note;
							note.instrumentId = instrumentId;
							note.pitch = (int)floorf(logf(freq_rel)*12.0f / logf(2.0f) + 0.5f) + pitchShift;
							note.start = pos;
							note.end = pos + fduration;
							m_notes.push_back(note);
						}
						pos += fduration;

						j++; // at freq1
						j++; // at freq2
						j++; // at next
					}
				}
			}
			else if (PyObject_TypeCheck(_item, &PyFloat_Type)) // note
			{

				float freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 0));
				int duration = (int)PyLong_AsLong(PyTuple_GetItem(item, 1));
				float fduration = (float)(duration * 60) / (float)(tempo * 48);

				if (freq_rel >0.0f)
				{
					VisNote note;
					note.instrumentId = instrumentId;
					note.pitch = (int)floorf(logf(freq_rel)*12.0f / logf(2.0f) + 0.5f) + pitchShift;
					note.start = pos;
					note.end = pos + fduration;
					m_notes.push_back(note);
				}
				pos += fduration;
			}
		}
	}		
}

void Visualizer::Play(unsigned bufferId) const
{
	TrackBuffer_deferred buffer = s_PyScoreDraft->GetTrackBuffer(bufferId);

	int argc = 0;
	char* argv = nullptr;
	QApplication app(argc, &argv);
	MainWidget widget(this, buffer);
	widget.show();
	app.exec();
}

typedef Deferred<Visualizer> Visualizer_deferred;
typedef std::vector<Visualizer_deferred> VisualizerMap;
static VisualizerMap s_visualizer_map;


static PyObject* InitVisualizer(PyObject *args)
{
	Visualizer_deferred visualizer;
	unsigned id = (unsigned)s_visualizer_map.size();
	s_visualizer_map.push_back(visualizer);
	return PyLong_FromUnsignedLong((unsigned long)(id));
}


static PyObject* DelVisualizer(PyObject *args)
{
	unsigned visualizerId = (unsigned)PyLong_AsUnsignedLong(args);
	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer.Abondon();

	return PyLong_FromLong(0);
}

static PyObject* ProcessNoteSeq(PyObject *args)
{
	unsigned visualizerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	unsigned instrumentId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	float startPosition = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 2));
	PyObject *seq_py = PyTuple_GetItem(args, 3);
	unsigned tempo = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 4));
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 5));

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer->ProcessNoteSeq(instrumentId, startPosition, seq_py, tempo, RefFreq);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* Play(PyObject *args)
{
	unsigned visualizerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	unsigned BufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer->Play(BufferId);

	return PyLong_FromUnsignedLong(0);
}


PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	s_PyScoreDraft = pyScoreDraft;

	pyScoreDraft->RegisterInterfaceExtension("MeteorInitVisualizer", InitVisualizer);
	pyScoreDraft->RegisterInterfaceExtension("MeteorDelVisualizer", DelVisualizer, "visualizerId", "visualizerId");
	pyScoreDraft->RegisterInterfaceExtension("MeteorProcessNoteSeq", ProcessNoteSeq, "visualizerId, instrument, startPos, seq, tempo, refFreq", "visualizerId, instrument.id, startPos, seq, tempo, refFreq");
	pyScoreDraft->RegisterInterfaceExtension("MeteorPlay", Play, "visualizerId, buffer", "visualizerId, buffer.id");
}

