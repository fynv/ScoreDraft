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
Percussion::Percussion() : m_beatVolume(1.0f)
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
	NoteBuffer beatBuf;
	beatBuf.m_sampleRate = (float)buffer.Rate();

	float fduration = (float)(duration * 60) / (float)(tempo * 48);
	float fNumOfSamples = buffer.Rate()*fduration;

	GenerateBeatWave(fNumOfSamples, &beatBuf);
	buffer.WriteBlend(beatBuf.m_sampleNum, beatBuf.m_data, fNumOfSamples, beatBuf.m_alignPos);
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

void Percussion::PlayBeats(TrackBuffer& buffer, Percussion_deferred* percussionList, const BeatSequence& seq, unsigned tempo)
{
	int i;
	int prog = 0;
	for (i = 0; i<(int)seq.size(); i++)
	{
		int newprog = (i + 1) * 10 / (int)seq.size();
		if (newprog>prog)
		{
			printf("-");
			prog = newprog;
		}
		const Beat& beat = seq[i];
	
		if (beat.m_PercId >= 0)
			percussionList[beat.m_PercId]->PlayBeat(buffer, beat.m_duration, tempo);
		else if (beat.m_duration >= 0)
			PlaySilence(buffer, beat.m_duration, tempo);
		else
			PlayBackspace(buffer, -beat.m_duration, tempo);
	}
	printf("\n");
}


bool Percussion::Tune(const char* cmd)
{
	char command[1024];
	sscanf(cmd, "%s", command);
	if (strcmp(command, "volume") == 0)
	{
		float value;
		if (sscanf(cmd + 7, "%f", &value))
			m_beatVolume = value;
		return true;
	}
	return false;
}
