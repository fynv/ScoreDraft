#include <Python.h>

#include "PyScoreDraft.h"
#include <stdio.h>
#include <cmath>
#include "Note.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define TIME_DIVISION 48

#include <vector>
#include "Deferred.h"

class NoteSequence;
typedef Deferred<NoteSequence> NoteSequence_deferred;
typedef std::vector<NoteSequence_deferred> SequenceList;

static void ByteSwap(unsigned short& word)
{
	unsigned short a=word&0xFF;
	a<<=8;
	word>>=8;
	word|=a;
}

static void ByteSwap(unsigned& word)
{
	unsigned a=word&0xFF;
	a<<=24;
	word>>=8;
	unsigned b=word&0xFF;
	b<<=16;
	word>>=8;
	unsigned c=word&0xFF;
	c<<=8;
	word|=a|b|c;
}

static void ToVariableLength(unsigned fixed, unsigned& variable,unsigned& bytes)
{
	variable=0;
	bytes=1;
	unsigned mask=0;
	while (fixed)
	{
		variable+=(fixed&0x7F)|mask;
		fixed>>=7;
		if (fixed) 
		{
			variable<<=8;
			bytes++;
			mask=0x80;
		}
	}

}

static void WriteBigEdianWord(FILE *fp, unsigned short aword)
{
	ByteSwap(aword);
	fwrite(&aword, 2, 1, fp);
}

static void WriteBigEdianDWord(FILE *fp, unsigned int adword)
{
	ByteSwap(adword);
	fwrite(&adword, 4, 1, fp);
}

static void WriteVariableLengthDWord(FILE *fp, unsigned fixeddword)
{
	unsigned variable;
	unsigned bytes;
	ToVariableLength(fixeddword,variable,bytes);
	fwrite(&variable, 1, bytes, fp);
}


struct NoteEvent
{
	bool isOn;
	unsigned time;
	unsigned char note;
};

#include <vector>

typedef std::vector<NoteEvent> NoteEventList;

static void AddNoteEvent(NoteEventList& list, const NoteEvent& nevent)
{
	if (list.empty() || (list.end() - 1)->time <= nevent.time)
	{
		list.push_back(nevent);
		return;
	}

	NoteEventList::iterator it = list.end() - 1;
	while (it != list.begin() && (it - 1)->time>nevent.time) it--;
	list.insert(it, nevent);

}

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

void WriteToMidi(const SequenceList& seqList, unsigned tempo, float refFreq, const char* fileName)
{
	size_t numOfTracks = seqList.size();

	FILE *fp = fopen(fileName, "wb");
	fwrite("MThd", 1, 4, fp);

	WriteBigEdianDWord(fp,6);
	WriteBigEdianWord(fp, 1);
	WriteBigEdianWord(fp, (unsigned short)numOfTracks);
	WriteBigEdianWord(fp, TIME_DIVISION);

	unsigned timeFactor = TIME_DIVISION / 48;

	//pitch shift
	float pitchShift = logf(refFreq / 261.626f)*12.0f / logf(2.0f) + 0.5f;

	unsigned maxTrackLength = 0;
	for (unsigned i = 0; i < numOfTracks; i++)
	{
		const NoteSequence& seq = *seqList[i];

		fwrite("MTrk", 1, 4, fp);
		WriteBigEdianDWord(fp,0); // trackLength;

		unsigned beginPoint = ftell(fp);

		unsigned char aByte;

		//Tempo Event	
		WriteVariableLengthDWord(fp, 0);
		aByte = 0xff;
		fwrite(&aByte, 1, 1, fp);
		aByte = 0x51;
		fwrite(&aByte, 1, 1, fp);
		aByte = 3;
		fwrite(&aByte, 1, 1, fp);

		unsigned theTempo = 60000000 / tempo;
		aByte = (theTempo & 0xff0000) >> 16;
		fwrite(&aByte, 1, 1, fp);
		aByte = (theTempo & 0xff00) >> 8;
		fwrite(&aByte, 1, 1, fp);
		aByte = theTempo & 0xff;
		fwrite(&aByte, 1, 1, fp);

		//Note Events
		NoteEventList elist;

		unsigned timeTicks = 0;
		unsigned j;
		for (j = 0; j <seq.size(); j++)
		{
			NoteEvent nevent;
			if (seq[j].m_freq_rel<0.0f)
				timeTicks += seq[j].m_duration*timeFactor;
			else if (seq[j].m_freq_rel>0.0f)
			{
				nevent.isOn = true;
				nevent.time = timeTicks;
				nevent.note = (unsigned char)(logf(seq[j].m_freq_rel)*12.0f / logf(2.0f)+ 60.0f + pitchShift);
				AddNoteEvent(elist, nevent);

				timeTicks += seq[j].m_duration*timeFactor;
				nevent.isOn = false;
				nevent.time = timeTicks;
				AddNoteEvent(elist, nevent);
			}
		}
		maxTrackLength = max(maxTrackLength, timeTicks);

		timeTicks = 0;
		for (j = 0; j<elist.size(); j++)
		{
			unsigned start = elist[j].time - timeTicks;
			timeTicks = elist[j].time;
			WriteVariableLengthDWord(fp,start);

			if (elist[j].isOn)
				aByte = 0x90;
			else
				aByte = 0x80;
			fwrite(&aByte, 1, 1, fp);

			aByte = elist[j].note;
			fwrite(&aByte, 1, 1, fp);

			aByte = 64;
			fwrite(&aByte, 1, 1, fp);
		}

		//// End of Track
		aByte = 0;
		fwrite(&aByte, 1, 1, fp);
		aByte = 0xff;
		fwrite(&aByte, 1, 1, fp);
		aByte = 0x2f;
		fwrite(&aByte, 1, 1, fp);
		aByte = 0;
		fwrite(&aByte, 1, 1, fp);

		//////////////////////////

		unsigned length = ftell(fp) - beginPoint;
		fseek(fp, beginPoint - 4, SEEK_SET);
		WriteBigEdianDWord(fp,length);
		fseek(fp, length, SEEK_CUR);
	}

	fclose(fp);
}

PyObject * WriteToMidi(PyObject *args)
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

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	pyScoreDraft->RegisterInterfaceExtension("WriteNoteSequencesToMidi", WriteToMidi,
		"seqList, tempo, refFreq, fileName", "seqList, tempo, refFreq, fileName", 
		"\t'''\n"
		"\tWrite a list of note sequences to a MIDI file.\n"
		"\tseqList -- a list of note sequences.\n"
		"\ttempo -- an integer indicating tempo in beats/minute.\n"
		"\trefFreq -- a float indicating reference frequency in Hz.\n"
		"\tfileName -- a string.\n"
		"\t'''\n");

}