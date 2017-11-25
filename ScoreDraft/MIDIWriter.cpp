#include "MIDIWriter.h"
#include <stdio.h>
#include <cmath>
#include "Note.h"
#include "parser/Document.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define TIME_DIVISION 48

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

void WriteToMidi(const Document& doc, const char* fileName)
{
	size_t numOfTracks = doc.m_tracks.size();
	unsigned tempo = doc.m_tempo;
	float refFreq = doc.m_RefFreq;

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
		const Track& track = doc.m_tracks[i];
		const NoteSequence& seq = *track.m_note_seq;

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
				nevent.note = (unsigned char)(logf(seq[j].m_freq_rel)*12.0f / logf(2.0f)+ 48.0f + pitchShift);
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

