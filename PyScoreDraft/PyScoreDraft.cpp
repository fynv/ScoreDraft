#include <Python.h>
#include <Note.h>
#include <Beat.h>
#include <Syllable.h>

#include <Instrument.h>
#include <Percussion.h>
#include <Singer.h>
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


static PyObject* CreateTrackBuffer(PyObject *self, PyObject *args)
{
	unsigned chn;
	if (!PyArg_ParseTuple(args, "I", &chn))
		return NULL;

	TrackBuffer* buffer= new TrackBuffer(44100,chn);
	return PyLong_FromVoidPtr(buffer);
}

static PyObject* DelTrackBuffer(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete buffer;
	return PyLong_FromLong(0);
}

static PyObject* TrackBufferSetVolume(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	buffer->SetVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* TrackBufferGetVolume(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyFloat_FromDouble((double)buffer->Volume());
}


static PyObject* TrackBufferSetPan(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float pan = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	buffer->SetPan(pan);
	return PyLong_FromLong(0);
}

static PyObject* TrackBufferGetPan(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyFloat_FromDouble((double)buffer->Pan());
}

static PyObject* TrackBufferGetNumberOfSamples(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)buffer->NumberOfSamples());
}

static PyObject* TrackBufferGetNumberOfChannels(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyLong_FromLong((long)buffer->NumberOfChannels());
}

static PyObject* TrackBufferGetCursor(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyFloat_FromDouble((double)buffer->GetCursor());
}

static PyObject* TrackBufferSetCursor(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float cursor = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	buffer->SetCursor(cursor);
	return PyLong_FromLong(0);
}

static PyObject* TrackBufferMoveCursor(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float cursor_delta = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	buffer->MoveCursor(cursor_delta);
	return PyLong_FromLong(0);
}

static PyObject* InstrumentPlay(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Instrument* instrument = (Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject *seq_py = PyTuple_GetItem(args, 2);
	PyObject *tempo_obj = PyTuple_GetItem(args, 3);
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));

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
	Instrument* instrument = (Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* cmd = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	instrument->Tune(cmd);
	return PyLong_FromLong(0);
}

static PyObject* InstrumentSetNoteVolume(PyObject *self, PyObject *args)
{
	Instrument* instrument = (Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	instrument->SetNoteVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* InstrumentSetNotePan(PyObject *self, PyObject *args)
{
	Instrument* instrument = (Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float pan = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	instrument->SetNotePan(pan);
	return PyLong_FromLong(0);
}

static PyObject* InstrumentIsGMDrum(PyObject *self, PyObject *args)
{
	Instrument* instrument = (Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	return PyBool_FromLong(instrument->IsGMDrum() ? 1 : 0);
}


static PyObject* PercussionPlay(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject *percId_list = PyTuple_GetItem(args, 1);
	PyObject *seq_py = PyTuple_GetItem(args, 2);
	PyObject *tempo_obj = PyTuple_GetItem(args, 3);

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
	Percussion **perc_List = new Percussion *[perc_count];
	for (size_t i = 0; i < perc_count; i++)
		perc_List[i] = (Percussion*)PyLong_AsVoidPtr(PyList_GetItem(percId_list, i));

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
	Percussion* perc = (Percussion*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* cmd = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	perc->Tune(cmd);
	return PyLong_FromLong(0);
}


static PyObject* PercussionSetBeatVolume(PyObject *self, PyObject *args)
{
	Percussion* perc = (Percussion*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	perc->SetBeatVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* PercussionSetBeatPan(PyObject *self, PyObject *args)
{
	Percussion* perc = (Percussion*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float pan = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	perc->SetBeatPan(pan);
	return PyLong_FromLong(0);
}

static PyObject* Sing(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	Singer* singer = (Singer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 1));
	PyObject *seq_py = PyTuple_GetItem(args, 2);

	PyObject *tempo_obj = PyTuple_GetItem(args, 3);
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 4));

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
	Singer* singer = (Singer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* cmd = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	singer->Tune(cmd);
	return PyLong_FromLong(0);
}

static PyObject* SingerSetDefaultLyric(PyObject *self, PyObject *args)
{
	Singer* singer = (Singer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* lyric = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	singer->SetDefaultLyric(lyric);
	return PyLong_FromLong(0);
}


static PyObject* SingerSetNoteVolume(PyObject *self, PyObject *args)
{
	Singer* singer = (Singer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float volume = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	singer->SetNoteVolume(volume);
	return PyLong_FromLong(0);
}

static PyObject* SingerSetNotePan(PyObject *self, PyObject *args)
{
	Singer* singer = (Singer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float pan = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	singer->SetNotePan(pan);
	return PyLong_FromLong(0);
}

static PyObject* MixTrackBufferList(PyObject *self, PyObject *args)
{
	TrackBuffer* targetBuffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	PyObject *list = PyTuple_GetItem(args, 1);
	
	size_t bufferCount = PyList_Size(list);
	TrackBuffer** bufferList = new TrackBuffer*[bufferCount];
	for (size_t i = 0; i < bufferCount; i++)
		bufferList[i] = (TrackBuffer*)PyLong_AsVoidPtr(PyList_GetItem(list, i));
	targetBuffer->CombineTracks((unsigned)bufferCount, bufferList);
	delete[] bufferList;

	return PyLong_FromUnsignedLong(0);
}

static PyObject* WriteTrackBufferToWav(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* fn = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	WriteToWav(*buffer, fn);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* ReadTrackBufferFromWav(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	const char* fn = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
	ReadFromWav(*buffer, fn);
	return PyLong_FromUnsignedLong(0);
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


static PyObject* CreateInternalInstrument(PyObject *self, PyObject *args)
{
	InternalInstrument_Type instrument_cls = (InternalInstrument_Type)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	Instrument* inst = nullptr;
	switch (instrument_cls)
	{
	case PureSin_Type:
		inst = new PureSin();
		break;
	case Square_Type:
		inst = new Square();
		break;
	case Triangle_Type:
		inst = new Triangle();
		break;
	case Sawtooth_Type:
		inst = new Sawtooth();
		break;
	case NaivePiano_Type:
		inst = new NaivePiano();
		break;
	case BottleBlow_Type:
		inst = new BottleBlow();
		break;
	}
	return PyLong_FromVoidPtr(inst);
}

static PyObject* DelInternalInstrument(PyObject *self, PyObject *args)
{
	Instrument* instrument = (Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete instrument;
	return PyLong_FromLong(0);
}

static PyMethodDef s_PyScoreDraftMethods[] = {
	{
		"CreateTrackBuffer",
		CreateTrackBuffer,
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
		"InstrumentIsGMDrum",
		InstrumentIsGMDrum,
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
		"TellDuration",
		TellDuration,
		METH_VARARGS,
		""
	},
	{
		"CreateInternalInstrument",
		CreateInternalInstrument,
		METH_VARARGS,
		""
	},
	{
		"DelInternalInstrument",
		DelInternalInstrument,
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
	s_PyScoreDraftMethods
}; 

PyMODINIT_FUNC PyInit_PyScoreDraft(void) {
	return PyModule_Create(&cModPyDem);
}
