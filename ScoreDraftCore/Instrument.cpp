#include "Instrument.h"
#include "Note.h"
#include "TrackBuffer.h"
#include <memory.h>
#include <cmath>
#include <vector>
#include <stdlib.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include <cmath>
#include <time.h>
Instrument::Instrument() : m_noteVolume(1.0f)
{
	srand((unsigned)time(NULL));
}

Instrument::~Instrument()
{

}

void Instrument::Silence(unsigned numOfSamples, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum=numOfSamples;
	noteBuf->Allocate();
	memset(noteBuf->m_data,0,sizeof(float)*numOfSamples);
}

void Instrument::GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	Silence((unsigned)ceilf(fNumOfSamples), noteBuf);
}

void Instrument::PlayNote(TrackBuffer& buffer, const Note& aNote, unsigned tempo, float RefFreq)
{
	NoteBuffer noteBuf;
	noteBuf.m_sampleRate = (float)buffer.Rate();

	float fduration=fabsf((float)(aNote.m_duration*60))/(float)(tempo*48);
	float fNumOfSamples = buffer.Rate()*fduration;

	if (aNote.m_freq_rel<0.0f)
	{
		if (aNote.m_duration>0) 
		{
			buffer.MoveCursor(fNumOfSamples);
			return;
		}
		else if (aNote.m_duration<0)
		{
			buffer.MoveCursor(-fNumOfSamples);
			return;
		}
		else return;
	}

	float freq = RefFreq*aNote.m_freq_rel;
	float sampleFreq=freq/(float)buffer.Rate();				
	GenerateNoteWave(fNumOfSamples, sampleFreq, &noteBuf);
	
	buffer.WriteBlend(noteBuf.m_sampleNum, noteBuf.m_data, fNumOfSamples, noteBuf.m_alignPos);
		
}

void Instrument::PlayNotes(TrackBuffer& buffer, const NoteSequence& seq, unsigned tempo, float RefFreq)
{
	int i;
	int prog=0;
	for (i=0;i<(int)seq.size();i++)
	{
		int newprog = (i + 1) * 10 / (int)seq.size();
		if (newprog>prog)
		{
			printf("-");
			prog=newprog;
		}
			
		PlayNote(buffer,seq[i],tempo,RefFreq);
	}
	printf("\n");
}

bool Instrument::Tune(const char* cmd)
{
	char command[1024];
	sscanf(cmd, "%s", command);
	if (strcmp(command, "volume") == 0)
	{
		float value;
		if (sscanf(cmd + 7, "%f", &value))
			m_noteVolume = value;
		return true;
	}
	return false;
}
