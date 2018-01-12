#include "Singer.h"
#include "SingingPiece.h"
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


VoiceBuffer::VoiceBuffer()
{
	m_sampleNum = 0;
	m_data = 0;
}

VoiceBuffer::~VoiceBuffer()
{
	delete[] m_data;
}

void VoiceBuffer::Allocate()
{
	delete[] m_data;
	m_data = new float[m_sampleNum];
}

Singer::Singer() : m_noteVolume(1.0f)
{

}

Singer::~Singer()
{

}


void Singer::Silence(unsigned numOfSamples, VoiceBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();
	memset(noteBuf->m_data, 0, sizeof(float)*numOfSamples);
}

void Singer::GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf)
{
	float totalDuration = 0.0f;
	for (size_t i = 0; i < notes.size(); i++)
		totalDuration += notes[i].fNumOfSamples;

	Silence((unsigned)ceilf(totalDuration), noteBuf);
}


void Singer::SingPiece(TrackBuffer& buffer, const SingingPiece& piece, unsigned tempo, float RefFreq)
{
	std::vector<SingerNoteParams> noteParams;
	VoiceBuffer noteBuf;
	float totalDuration = 0.0f;

	for (size_t i = 0; i < piece.m_notes.size(); i++)
	{
		const Note& aNote = piece.m_notes[i];
		float fduration = fabsf((float)(aNote.m_duration * 60)) / (float)(tempo * 48);
		float fNumOfSamples = buffer.Rate()*fduration;
		if (aNote.m_freq_rel<0.0f)
		{
			if (noteParams.size()>0)
			{
				std::string lyric = piece.m_lyric;
				if (lyric == "") lyric = m_defaultLyric;
				GenerateWave(lyric.data(), noteParams, &noteBuf);
				buffer.WriteBlend(noteBuf.m_sampleNum, noteBuf.m_data, totalDuration);
				noteParams.clear();
				totalDuration = 0.0f;
			}

			if (aNote.m_duration>0)
			{
				buffer.MoveCursor(fNumOfSamples);
			}
			else if (aNote.m_duration<0)
			{
				buffer.MoveCursor(-fNumOfSamples);
			}
			continue;
		}

		SingerNoteParams param;
		float freq = RefFreq*aNote.m_freq_rel;
		param.sampleFreq = freq / (float)buffer.Rate();
		param.fNumOfSamples = fNumOfSamples;
		noteParams.push_back(param);
		totalDuration += fNumOfSamples;
	}

	if (noteParams.size()>0)
	{
		std::string lyric = piece.m_lyric;
		if (lyric == "") lyric = m_defaultLyric;
		GenerateWave(lyric.data(), noteParams, &noteBuf);
		buffer.WriteBlend(noteBuf.m_sampleNum, noteBuf.m_data, totalDuration);
	}

}

void Singer::SingSequence(TrackBuffer& buffer, const SingingSequence& seq, unsigned tempo, float RefFreq)
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

		SingPiece(buffer, seq[i], tempo, RefFreq);
	}
	printf("\n");
}

bool Singer::Tune(const char* cmd)
{
	char command[1024];
	sscanf(cmd, "%s", command);
	if (strcmp(command, "volume") == 0)
	{
		float value;
		sscanf(cmd+7, "%f", &value);
		m_noteVolume = value;
		return true;
	}
	else if (strcmp(command, "default_lyric") == 0)
	{
		char lyric[1024];
		sscanf(cmd + 14, "%s", lyric);
		m_defaultLyric = lyric;
	}
	return false;
}
