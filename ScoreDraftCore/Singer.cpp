#include "Singer.h"
#include "Syllable.h"
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

Singer::Singer() : m_noteVolume(1.0f), m_notePan(0.0f)
{
	m_lyric_charset = "utf-8";
}

Singer::~Singer()
{

}


void Singer::Silence(unsigned numOfSamples, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();
	memset(noteBuf->m_data, 0, sizeof(float)*numOfSamples);
}

void Singer::GenerateWave(SyllableInternal syllable, NoteBuffer* noteBuf)
{
	float totalDuration = syllable.GetTotalDuration();
	Silence((unsigned)ceilf(totalDuration), noteBuf);
}

void Singer::GenerateWave_SingConsecutive(SyllableInternalList syllableList, NoteBuffer* noteBuf)
{
	float totalDuration = 0.0f;
	for (size_t j = 0; j < syllableList.size(); j++)
	{
		SyllableInternal& syllable = *syllableList[j];
		totalDuration += syllable.GetTotalDuration();
	}
	Silence((unsigned)ceilf(totalDuration), noteBuf);
}

void Singer::SingSyllable(TrackBuffer& buffer, const Syllable& syllable, unsigned tempo, float RefFreq)
{
	std::vector<ControlPointInternal> ctrlpnts;

	float totalDuration = 0.0f;

	for (size_t i = 0; i < syllable.m_ctrlPnts.size(); i++)
	{
		const ControlPoint& aCtrlPnt = syllable.m_ctrlPnts[i];
		float fduration = fabsf((float)(aCtrlPnt.m_duration * 60)) / (float)(tempo * 48);
		float fNumOfSamples = buffer.Rate()*fduration;
		if (aCtrlPnt.m_freq_rel<0.0f)
		{
			if (ctrlpnts.size()>0)
			{
				std::string lyric = syllable.m_lyric;
				if (lyric == "") lyric = m_defaultLyric;
				SyllableInternal _syllable;
				_syllable.lyric = lyric;
				_syllable.ctrlPnts = ctrlpnts;

				NoteBuffer noteBuf;
				noteBuf.m_sampleRate = (float)buffer.Rate();
				noteBuf.m_cursorDelta = totalDuration;
				noteBuf.m_volume = m_noteVolume;
				noteBuf.m_pan = m_notePan;

				GenerateWave(_syllable, &noteBuf);
				buffer.WriteBlend(noteBuf);
				ctrlpnts.clear();
				totalDuration = 0.0f;
			}

			if (aCtrlPnt.m_duration>0)
			{
				buffer.MoveCursor(fNumOfSamples);
			}
			else if (aCtrlPnt.m_duration<0)
			{
				buffer.MoveCursor(-fNumOfSamples);
			}
			continue;
		}

		ControlPointInternal ctrlpnt;
		float freq = RefFreq*aCtrlPnt.m_freq_rel;
		ctrlpnt.sampleFreq = freq / (float)buffer.Rate();
		ctrlpnt.fNumOfSamples = fNumOfSamples;
		ctrlpnts.push_back(ctrlpnt);
		totalDuration += fNumOfSamples;
	}

	if (ctrlpnts.size()>0)
	{
		std::string lyric = syllable.m_lyric;
		if (lyric == "") lyric = m_defaultLyric;
		SyllableInternal _syllable;
		_syllable.lyric = lyric;
		_syllable.ctrlPnts = ctrlpnts;
		NoteBuffer noteBuf;
		noteBuf.m_sampleRate = (float)buffer.Rate();
		noteBuf.m_cursorDelta = totalDuration;
		noteBuf.m_volume = m_noteVolume;
		noteBuf.m_pan = m_notePan;

		GenerateWave(_syllable, &noteBuf);
		buffer.WriteBlend(noteBuf);
	}

}

void Singer::SingConsecutiveSyllables(TrackBuffer& buffer, const SyllableSequence& syllables, unsigned tempo, float RefFreq)
{
	SyllableInternalList syllableList;

	float totalDuration = 0.0f;

	for (size_t j = 0; j < syllables.size(); j++)
	{
		const Syllable& syllable = syllables[j];
		std::vector<ControlPointInternal> ctrlpnts;

		for (size_t i = 0; i < syllable.m_ctrlPnts.size(); i++)
		{
			const ControlPoint& aCtrlPnt = syllable.m_ctrlPnts[i];
			float fduration = fabsf((float)(aCtrlPnt.m_duration * 60)) / (float)(tempo * 48);
			float fNumOfSamples = buffer.Rate()*fduration;
			if (aCtrlPnt.m_freq_rel < 0.0f)
			{
				if (syllableList.size()>0 || ctrlpnts.size()>0)
				{
					if (ctrlpnts.size() > 0)
					{
						std::string lyric = syllable.m_lyric;
						if (lyric == "") lyric = m_defaultLyric;
						SyllableInternal_Deferred _syllable;
						_syllable->lyric = lyric;
						_syllable->ctrlPnts = ctrlpnts;
						syllableList.push_back(_syllable);
					}
					NoteBuffer noteBuf;
					noteBuf.m_sampleRate = (float)buffer.Rate();
					noteBuf.m_cursorDelta = totalDuration;
					noteBuf.m_volume = m_noteVolume;
					noteBuf.m_pan = m_notePan;

					GenerateWave_SingConsecutive(syllableList, &noteBuf);
					buffer.WriteBlend(noteBuf);
					ctrlpnts.clear();
					syllableList.clear();
					totalDuration = 0.0f;
				}

				if (aCtrlPnt.m_duration>0)
				{
					buffer.MoveCursor(fNumOfSamples);
				}
				else if (aCtrlPnt.m_duration<0)
				{
					buffer.MoveCursor(-fNumOfSamples);
				}
				continue;
			}
			ControlPointInternal ctrlpnt;
			float freq = RefFreq*aCtrlPnt.m_freq_rel;
			ctrlpnt.sampleFreq = freq / (float)buffer.Rate();
			ctrlpnt.fNumOfSamples = fNumOfSamples;
			ctrlpnts.push_back(ctrlpnt);
			totalDuration += fNumOfSamples;
		}
		if (ctrlpnts.size()>0)
		{
			std::string lyric = syllable.m_lyric;
			if (lyric == "") lyric = m_defaultLyric;
			SyllableInternal_Deferred _syllable;
			_syllable->lyric = lyric;
			_syllable->ctrlPnts = ctrlpnts;
			syllableList.push_back(_syllable);
		}		
	}

	if (syllableList.size() > 0)
	{
		NoteBuffer noteBuf;
		noteBuf.m_sampleRate = (float)buffer.Rate();
		noteBuf.m_cursorDelta = totalDuration;
		noteBuf.m_volume = m_noteVolume;
		noteBuf.m_pan = m_notePan;

		GenerateWave_SingConsecutive(syllableList, &noteBuf);
		buffer.WriteBlend(noteBuf);
	}
}


bool Singer::Tune(const char* cmd)
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
	else if (strcmp(command, "default_lyric") == 0)
	{
		char lyric[1024];
		if (sscanf(cmd + 14, "%s", lyric))
			m_defaultLyric = lyric;
		return true;
	}
	return false;
}
