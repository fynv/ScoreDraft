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
Instrument::Instrument() : m_noteVolume(1.0f), m_notePan(0.0f)
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

	NoteBuffer noteBuf;
	noteBuf.m_sampleRate = (float)buffer.Rate();
	noteBuf.m_cursorDelta = fNumOfSamples;
	noteBuf.m_volume = m_noteVolume;
	noteBuf.m_pan = m_notePan;

	GenerateNoteWave(fNumOfSamples, sampleFreq, &noteBuf);
	
	buffer.WriteBlend(noteBuf);
		
}

bool Instrument::Tune(const char* cmd)
{
	char command[1024];
	sscanf(cmd, "%s", command);
	if (strcmp(command, "volume") == 0)
	{
		float value;
		if (sscanf(cmd + 7, "%f", &value))
		{
			if (value < 0.0f) value = 0.0f;
			m_noteVolume = value;
		}
		return true;
	}
	else if (strcmp(command, "pan") == 0)
	{
		float value;
		if (sscanf(cmd + 4, "%f", &value))
		{
			if (value<-1.0f) value = -1.0f;
			else if (value>1.0f) value = 1.0f;
			m_notePan = value;
		}
		return true;
	}
	return false;
}
