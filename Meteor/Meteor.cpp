#include "PyScoreDraft.h"
#include "Deferred.h"
#include <string.h>
#include "Meteor.h"
#include "MainWidget.h"
#include <qapplication.h>

static PyScoreDraft* s_PyScoreDraft;

void Visualizer::ProcessNoteSeq(unsigned instrumentId, float startPosition, float sampleRate, PyObject *seq_py, unsigned tempo, float RefFreq, TempoMap *tempoMap)
{
	bool tempo_map = tempoMap != nullptr;

	float pos = startPosition;
	int pitchShift = (int)floorf(logf(RefFreq / 261.626f)*12.0f / logf(2.0f) + 0.5f);

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

							float freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(_item, 0));
							int duration = (int)PyLong_AsLong(PyTuple_GetItem(_item, 1));

							float fduration; 
							if (tempo_map)
							{
								int pos1 = beatPos;
								int pos2 = pos1 + duration;
								float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
								fduration = fNumOfSamples / sampleRate;
							}
							else
							{
								fduration = (float)(duration * 60) / (float)(tempo * 48);
							}

							beatPos += duration;

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

						float fduration;
						if (tempo_map)
						{
							int pos1 = beatPos;
							int pos2 = pos1 + duration;
							float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
							fduration = fNumOfSamples / sampleRate;
						}
						else
						{
							fduration = (float)(duration * 60) / (float)(tempo * 48);
						}

						beatPos += duration;

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

				float fduration;
				if (tempo_map)
				{
					int pos1 = beatPos;
					int pos2 = pos1 + duration;
					float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
					fduration = fNumOfSamples / sampleRate;
				}
				else
				{
					fduration = (float)(duration * 60) / (float)(tempo * 48);
				}

				beatPos += duration;

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
	m_needUpdateSublists = true;
}

void Visualizer::ProcessBeatSeq(unsigned percIdList[], float startPosition, float sampleRate, PyObject *seq_py, unsigned tempo, TempoMap *tempoMap)
{
	bool tempo_map = tempoMap != nullptr;

	float pos = startPosition;
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

			float fduration;
			if (tempo_map)
			{
				int pos1 = beatPos;
				int pos2 = pos1 + duration;
				float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
				fduration = fNumOfSamples / sampleRate;
			}
			else
			{
				fduration = (float)(duration * 60) / (float)(tempo * 48);
			}
			beatPos += duration;

			if (percId >= 0)
			{
				VisBeat beat;
				beat.percId = percId;
				beat.start = pos;
				beat.end = pos + fduration;
				m_beats.push_back(beat);
			}
			pos += fduration;
		}
	}
	m_needUpdateSublists = true;
}

float VisSinging::CreateFromSyllable(unsigned singerId, float pos, float sampleRate, float pitchShift, unsigned tempo, const Syllable& syllable, TempoMap *tempoMap, int tempoMapOffset)
{
	bool tempo_map = tempoMap != nullptr;

	this->singerId = singerId;
	this->lyric = syllable.m_lyric;
	this->start = pos;

	int iTotalLen = 0;
	for (unsigned i = 0; i < (unsigned)syllable.m_ctrlPnts.size(); i++)
	{
		int iDuration = syllable.m_ctrlPnts[i].m_duration;
		iTotalLen += iDuration;
	}

	float totalLen;
	if (tempo_map)
	{
		int pos1 = tempoMapOffset;
		int pos2 = pos1 + iTotalLen;
		float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
		totalLen = fNumOfSamples / sampleRate;
	}
	else
	{
		totalLen = (float)(iTotalLen * 60) / (float)(tempo * 48);
	}


	this->end = pos + totalLen;

	float pitchPerSec = 50.0f;
	float secPerPitchSample = 1.0f / pitchPerSec;

	unsigned pitchLen = (unsigned)ceilf(totalLen*pitchPerSec);
	this->pitch.resize(pitchLen);
	unsigned pitchPos = 0;
	float fPitchPos = 0.0f;
	float nextPitchPos=0.0f;

	int beatPos = tempoMapOffset;
	for (unsigned i = 0; i < (unsigned)syllable.m_ctrlPnts.size(); i++)
	{
		int iDuration = syllable.m_ctrlPnts[i].m_duration;
		if (iDuration <= 0) continue;

		float fduration;
		if (tempo_map)
		{
			int pos1 = beatPos;
			int pos2 = pos1 + iDuration;
			float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
			fduration = fNumOfSamples / sampleRate;
			beatPos = pos2;
		}
		else
		{
			fduration = (float)(iDuration * 60) / (float)(tempo * 48);
		}
		float freq1 = syllable.m_ctrlPnts[i].m_freq_rel;
		float freq2 = i < syllable.m_ctrlPnts.size() - 1 ? syllable.m_ctrlPnts[i + 1].m_freq_rel : freq1;
		float thisPitchPos = nextPitchPos;
		nextPitchPos += fduration;
		for (; fPitchPos < nextPitchPos && pitchPos<pitchLen; fPitchPos += secPerPitchSample, pitchPos++)
		{
			float k = (fPitchPos - thisPitchPos) / fduration;
			pitch[pitchPos] = freq1*(1.0f-k)+freq2*k;
		}
	}

	// smoothing
	float* temp = new float[pitchLen];
	temp[0] = pitch[0];
	temp[pitchLen - 1] = pitch[pitchLen - 1];
	for (unsigned i = 1; i < pitchLen - 1; i++)
	{
		temp[i] = 0.25f*(pitch[i - 1] + pitch[i + 1]) + 0.5f*pitch[i];
	}

	for (unsigned i = 0; i < pitchLen; i++)
	{
		pitch[i] = logf(temp[i])*12.0f / logf(2.0f) + pitchShift;
	}

	delete[] temp;

	return totalLen;
}


void Visualizer::ProcessSingingSeq(unsigned singerId, float startPosition, float sampleRate, PyObject *seq_py, unsigned tempo, float RefFreq, TempoMap *tempoMap)
{
	bool tempo_map = tempoMap != nullptr;

	float pos = startPosition;
	float pitchShift = floorf(logf(RefFreq / 261.626f)*12.0f / logf(2.0f) + 0.5f);

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
				size_t tupleSize = PyTuple_Size(item);

				size_t j = 0;
				while (j < tupleSize)
				{
					_item = PyTuple_GetItem(item, j);
					std::string lyric = _PyUnicode_AsString(_item);
					j++;

					_item = PyTuple_GetItem(item, j);
					if (PyObject_TypeCheck(_item, &PyTuple_Type)) // singing note
					{
						int iTotalDuration = 0;

						Syllable syllable;
						syllable.m_lyric = lyric;
						for (; j<tupleSize; j++)
						{
							_item = PyTuple_GetItem(item, j);
							if (!PyObject_TypeCheck(_item, &PyTuple_Type)) break;

							unsigned numCtrlPnt = (unsigned)(PyTuple_Size(_item) + 1) / 2;
							for (unsigned k = 0; k < numCtrlPnt; k++)
							{
								ControlPoint ctrlPnt;
								ctrlPnt.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(_item, k * 2));
								if (k * 2 + 1 < PyTuple_Size(_item))
								{
									ctrlPnt.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(_item, k * 2 + 1));
									iTotalDuration += ctrlPnt.m_duration;
								}
								else
									ctrlPnt.m_duration = 0;

								if (ctrlPnt.m_freq_rel > 0.0f)
								{
									syllable.m_ctrlPnts.push_back(ctrlPnt);
								}
								else
								{
									if (syllable.m_ctrlPnts.size() > 0)
									{
										VisSinging singing;
										float syllableLen= singing.CreateFromSyllable(singerId, pos, sampleRate, pitchShift, tempo, syllable, tempoMap, beatPos);
										m_singings.push_back(singing);
										syllable.m_ctrlPnts.clear();
										pos += syllableLen;
										beatPos += iTotalDuration;
										iTotalDuration = 0;
									}
									float fduration;
									if (tempo_map)
									{
										int pos1 = beatPos;
										int pos2 = pos1 + ctrlPnt.m_duration;
										float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
										fduration = fNumOfSamples / sampleRate;									
									}
									else
									{
										fduration = (float)(ctrlPnt.m_duration * 60) / (float)(tempo * 48);
									}
									pos += fduration;
								}
							}
							if (syllable.m_ctrlPnts.size() > 0)
							{
								ControlPoint& lastCtrlPnt = *(syllable.m_ctrlPnts.end() - 1);
								if (lastCtrlPnt.m_freq_rel > 0.0f && lastCtrlPnt.m_duration > 0)
								{
									ControlPoint ctrlPnt;
									ctrlPnt.m_freq_rel = lastCtrlPnt.m_freq_rel;
									ctrlPnt.m_duration = 0;
									syllable.m_ctrlPnts.push_back(ctrlPnt);
								}
							}
						}
						if (syllable.m_ctrlPnts.size() > 0)
						{
							VisSinging singing;
							float syllableLen = singing.CreateFromSyllable(singerId, pos, sampleRate, pitchShift, tempo, syllable, tempoMap, beatPos);
							m_singings.push_back(singing);
							pos += syllableLen;
							beatPos += iTotalDuration;
							iTotalDuration = 0;
						}			
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

						float fduration;
						if (freq1 > 0.0 && freq2 > 0.0)
						{
							ControlPoint ctrlPnt;
							ctrlPnt.m_duration = duration;
							ctrlPnt.m_freq_rel = freq1;
							syllable.m_ctrlPnts.push_back(ctrlPnt);
							ctrlPnt.m_duration = 0;
							ctrlPnt.m_freq_rel = freq2;
							syllable.m_ctrlPnts.push_back(ctrlPnt);

							VisSinging singing;
							fduration = singing.CreateFromSyllable(singerId, pos, sampleRate, pitchShift, tempo, syllable, tempoMap, beatPos);
							m_singings.push_back(singing);
						}
						else
						{
							if (tempo_map)
							{
								int pos1 = beatPos;
								int pos2 = pos1 + duration;
								float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
								fduration = fNumOfSamples / sampleRate;
							}
							else
							{
								fduration = (float)(duration * 60) / (float)(tempo * 48);
							}
							
						}
						beatPos += duration;
						pos += fduration;
					}

				}
			}
			else if (PyObject_TypeCheck(_item, &PyFloat_Type)) // note
			{
				Syllable syllable;
				syllable.m_lyric = "";

				ControlPoint ctrlPnt;
				ctrlPnt.m_freq_rel = (float)PyFloat_AsDouble(PyTuple_GetItem(item, 0));
				ctrlPnt.m_duration = (int)PyLong_AsLong(PyTuple_GetItem(item, 1));
				syllable.m_ctrlPnts.push_back(ctrlPnt);

				float fduration;

				if (ctrlPnt.m_freq_rel > 0.0f)
				{
					ctrlPnt.m_duration = 0;
					syllable.m_ctrlPnts.push_back(ctrlPnt);

					VisSinging singing;
					fduration = singing.CreateFromSyllable(singerId, pos, sampleRate, pitchShift, tempo, syllable, tempoMap, beatPos);
					m_singings.push_back(singing);
				}
				else
				{
					if (tempo_map)
					{
						int pos1 = beatPos;
						int pos2 = pos1 + ctrlPnt.m_duration;
						float fNumOfSamples = GetTempoMap(*tempoMap, pos2) - GetTempoMap(*tempoMap, pos1);
						fduration = fNumOfSamples / sampleRate;
					}
					else
					{
						fduration = (float)(ctrlPnt.m_duration * 60) / (float)(tempo * 48);
					}
				}

				beatPos += ctrlPnt.m_duration;
				pos += fduration;
			}

		}
	}
	m_needUpdateSublists = true;
}

void Visualizer::_updateSublists()
{
	m_notes_sublists.SetData(m_notes, 3.0f);
	m_beats_sublists.SetData(m_beats, 3.0f);
	m_singing_sublists.SetData(m_singings, 3.0f);
	m_needUpdateSublists = false;
}

void Visualizer::Play(unsigned bufferId) 
{
	if (m_needUpdateSublists)
		_updateSublists();

	TrackBuffer_deferred buffer = s_PyScoreDraft->GetTrackBuffer(bufferId);

	int argc = 0;
	char* argv = nullptr;
	QApplication app(argc, &argv);

	QSurfaceFormat fmt;
	fmt.setSamples(8);
	QSurfaceFormat::setDefaultFormat(fmt);

	MainWidget widget(this, buffer);
	widget.show();
	app.exec();
}

void Visualizer::SaveToFile(const char* filename)
{
	if (m_needUpdateSublists)
		_updateSublists();

	FILE *fp=fopen(filename, "wb");
	unsigned countNotes = (unsigned) m_notes.size();
	fwrite(&countNotes, sizeof(unsigned), 1, fp);
	fwrite(&m_notes[0], sizeof(VisNote), countNotes, fp);
	m_notes_sublists.SaveToFile(fp);
	unsigned countBeats = (unsigned)m_beats.size();
	fwrite(&countBeats, sizeof(unsigned), 1, fp);
	fwrite(&m_beats[0], sizeof(VisBeat), countBeats, fp);
	m_beats_sublists.SaveToFile(fp);
	unsigned countSinging = (unsigned)m_singings.size();
	fwrite(&countSinging, sizeof(unsigned), 1, fp);
	for (unsigned i = 0; i < countSinging; i++)
	{
		const VisSinging& singing = m_singings[i];
		fwrite(&singing.singerId, sizeof(unsigned), 1, fp);
		unsigned char len = (unsigned char)singing.lyric.length();
		fwrite(&len, 1, 1, fp);
		if (len>0)
			fwrite(singing.lyric.data(), 1, len, fp);
		unsigned count = (unsigned)singing.pitch.size();
		fwrite(&count, sizeof(unsigned), 1, fp);
		fwrite(singing.pitch.data(), sizeof(float), count, fp);
		fwrite(&singing.start, sizeof(float), 2, fp);
	}
	m_singing_sublists.SaveToFile(fp);
	fclose(fp);
}

void Visualizer::LoadFromFile(const char* filename)
{
	FILE *fp = fopen(filename, "rb");
	unsigned countNotes;
	fread(&countNotes, sizeof(unsigned), 1, fp);
	m_notes.clear();
	m_notes.resize(countNotes);
	fread(&m_notes[0], sizeof(VisNote), countNotes, fp);
	m_notes_sublists.LoadFromFile(fp);
	unsigned countBeats;
	fread(&countBeats, sizeof(unsigned), 1, fp);
	m_beats.clear();
	m_beats.resize(countBeats);
	fread(&m_beats[0], sizeof(VisBeat), countBeats, fp);
	m_beats_sublists.LoadFromFile(fp);
	unsigned countSinging;
	fread(&countSinging, sizeof(unsigned), 1, fp);
	m_singings.clear();
	m_singings.resize(countSinging);
	for (unsigned i = 0; i < countSinging; i++)
	{
		VisSinging& singing = m_singings[i];
		fread(&singing.singerId, sizeof(unsigned), 1, fp);
		unsigned char len;
		fread(&len, 1, 1, fp);
		if (len > 0)
		{
			char _str[256];
			fread(_str, 1, len, fp);
			_str[len] = 0;
			singing.lyric = _str;
		}
		else
			singing.lyric = "";
		unsigned count;
		fread(&count, sizeof(unsigned), 1, fp);
		singing.pitch.resize(count);
		fread(singing.pitch.data(), sizeof(float), count, fp);
		fread(&singing.start, sizeof(float), 2, fp);
	}
	m_singing_sublists.LoadFromFile(fp);
	fclose(fp);
	m_needUpdateSublists = true;
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

	unsigned bufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	TrackBuffer_deferred buffer = s_PyScoreDraft->GetTrackBuffer(bufferId);
	float sampleRate = buffer->Rate();
	float startPosition = buffer->GetCursor() / sampleRate;

	PyObject *seq_py = PyTuple_GetItem(args, 3);
	PyObject *tempo_obj = PyTuple_GetItem(args, 4);
	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 5));

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

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer->ProcessNoteSeq(instrumentId, startPosition, sampleRate, seq_py, tempo, RefFreq, tempo_map? &tempoMap: nullptr);

	return PyLong_FromUnsignedLong(0);
}

static PyObject* ProcessBeatSeq(PyObject *args)
{
	unsigned visualizerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	PyObject *percId_list = PyTuple_GetItem(args, 1);

	unsigned bufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	TrackBuffer_deferred buffer = s_PyScoreDraft->GetTrackBuffer(bufferId);
	float sampleRate = buffer->Rate();
	float startPosition = buffer->GetCursor() / sampleRate;

	PyObject *seq_py = PyTuple_GetItem(args, 3);
	PyObject *tempo_obj = PyTuple_GetItem(args, 4);

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

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];

	size_t perc_count = PyList_Size(percId_list);
	unsigned *percIdList = new unsigned[perc_count];
	for (size_t i = 0; i < perc_count; i++)
		percIdList[i]= (unsigned)PyLong_AsUnsignedLong(PyList_GetItem(percId_list, i));

	visualizer->ProcessBeatSeq(percIdList, startPosition, sampleRate, seq_py, tempo, tempo_map ? &tempoMap : nullptr);
	
	delete[] percIdList;

	return PyLong_FromUnsignedLong(0);

}


static PyObject* ProcessSingingSeq(PyObject *args)
{
	unsigned visualizerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	unsigned singerId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));

	unsigned bufferId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 2));
	TrackBuffer_deferred buffer = s_PyScoreDraft->GetTrackBuffer(bufferId);
	float sampleRate = buffer->Rate();
	float startPosition = buffer->GetCursor() / sampleRate;

	PyObject *seq_py = PyTuple_GetItem(args, 3);
	PyObject *tempo_obj = PyTuple_GetItem(args, 4);

	float RefFreq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 5));

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

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer->ProcessSingingSeq(singerId, startPosition, sampleRate, seq_py, tempo, RefFreq, tempo_map ? &tempoMap : nullptr);

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

static PyObject* SaveToFile(PyObject *args)
{
	unsigned visualizerId;
	const char* fn;
	if (!PyArg_ParseTuple(args, "Is", &visualizerId, &fn))
		return NULL;

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer->SaveToFile(fn);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* LoadFromFile(PyObject *args)
{
	unsigned visualizerId;
	const char* fn;
	if (!PyArg_ParseTuple(args, "Is", &visualizerId, &fn))
		return NULL;

	Visualizer_deferred visualizer = s_visualizer_map[visualizerId];
	visualizer->LoadFromFile(fn);
	return PyLong_FromUnsignedLong(0);
}

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	s_PyScoreDraft = pyScoreDraft;

	pyScoreDraft->RegisterInterfaceExtension("MeteorInitVisualizer", InitVisualizer);
	pyScoreDraft->RegisterInterfaceExtension("MeteorDelVisualizer", DelVisualizer, "visualizerId", "visualizerId");
	pyScoreDraft->RegisterInterfaceExtension("MeteorProcessNoteSeq", ProcessNoteSeq, "visualizerId, instrument, buf, seq, tempo, refFreq", "visualizerId, instrument.id, buf.id, seq, tempo, refFreq");
	pyScoreDraft->RegisterInterfaceExtension("MeteorProcessBeatSeq", ProcessBeatSeq, "visualizerId, percList, buf, seq, tempo", "visualizerId, ObjectToId(percList), buf.id, seq, tempo");
	pyScoreDraft->RegisterInterfaceExtension("MeteorProcessSingingSeq", ProcessSingingSeq, "visualizerId, singer, buf, seq, tempo, refFreq", "visualizerId, singer.id, buf.id, seq, tempo, refFreq");
	pyScoreDraft->RegisterInterfaceExtension("MeteorPlay", Play, "visualizerId, buffer", "visualizerId, buffer.id");
	pyScoreDraft->RegisterInterfaceExtension("MeteorSaveToFile", SaveToFile, "visualizerId, filename", "visualizerId, filename");
	pyScoreDraft->RegisterInterfaceExtension("MeteorLoadFromFile", LoadFromFile, "visualizerId, filename", "visualizerId, filename");
}

