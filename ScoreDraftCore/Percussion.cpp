#include "Percussion.h"
#include "Beat.h"
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
Percussion::Percussion() : m_beatVolume(1.0f), m_beatPan(0.0f)
{
	srand((unsigned)time(NULL));
}

Percussion::~Percussion()
{

}


void Percussion::Silence(unsigned numOfSamples, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();
	memset(noteBuf->m_data, 0, sizeof(float)*numOfSamples);
}

void Percussion::GenerateBeatWave(float fNumOfSamples, NoteBuffer* beatBuf)
{
	Silence((unsigned)ceilf(fNumOfSamples), beatBuf);
}

void Percussion::PlayBeat(TrackBuffer& buffer, int duration, unsigned tempo)
{

	float fduration = (float)(duration * 60) / (float)(tempo * 48);
	float fNumOfSamples = buffer.Rate()*fduration;

	NoteBuffer beatBuf;
	beatBuf.m_sampleRate = (float)buffer.Rate();
	beatBuf.m_cursorDelta = fNumOfSamples;
	beatBuf.m_volume = m_beatVolume;
	beatBuf.m_pan = m_beatPan;

	GenerateBeatWave(fNumOfSamples, &beatBuf);
	buffer.WriteBlend(beatBuf);
}

void Percussion::PlaySilence(TrackBuffer& buffer, int duration, unsigned tempo)
{
	float fduration = (float)(duration * 60) / (float)(tempo * 48);
	float fNumOfSamples = buffer.Rate()*fduration;
	buffer.MoveCursor(fNumOfSamples);
}

void Percussion::PlayBackspace(TrackBuffer& buffer, int duration, unsigned tempo)
{
	float fduration = (float)(duration * 60) / (float)(tempo * 48);
	float fNumOfSamples = buffer.Rate()*fduration;
	buffer.MoveCursor(-fNumOfSamples);
	return;
}

bool Percussion::Tune(const char* cmd)
{
	char command[1024];
	sscanf(cmd, "%s", command);
	if (strcmp(command, "volume") == 0)
	{
		float value;
		if (sscanf(cmd + 7, "%f", &value))
		{
			if (value < 0.0f) value = 0.0f;
			m_beatVolume = value;
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
			m_beatPan = value;
		}
		return true;
	}
	return false;
}
