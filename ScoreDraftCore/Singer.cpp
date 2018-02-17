#include "Singer.h"
#include "SingingPiece.h"
#include "RapPiece.h"
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

Singer::Singer() : m_noteVolume(1.0f)
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

void Singer::GenerateWave(SingingPieceInternal piece, NoteBuffer* noteBuf)
{
	float totalDuration = 0.0f;
	for (size_t i = 0; i < piece.notes.size(); i++)
		totalDuration += piece.notes[i].fNumOfSamples;

	Silence((unsigned)ceilf(totalDuration), noteBuf);
}

void Singer::GenerateWave_Rap(RapPieceInternal piece, NoteBuffer* noteBuf)
{
	Silence((unsigned)ceilf(piece.fNumOfSamples), noteBuf);
}

void Singer::GenerateWave_SingConsecutive(SingingPieceInternalList pieceList, NoteBuffer* noteBuf)
{
	float totalDuration = 0.0f;
	for (size_t j = 0; j < pieceList.size(); j++)
	{
		SingingPieceInternal& piece = *pieceList[j];
		for (size_t i = 0; i < piece.notes.size(); i++)
			totalDuration += piece.notes[i].fNumOfSamples;
	}
	Silence((unsigned)ceilf(totalDuration), noteBuf);

}

void Singer::GenerateWave_RapConsecutive(RapPieceInternalList pieceList, NoteBuffer* noteBuf)
{
	float totalDuration = 0.0f;
	for (size_t j = 0; j < pieceList.size(); j++)
	{
		RapPieceInternal& piece = *pieceList[j];
		totalDuration += piece.fNumOfSamples;
	}
	Silence((unsigned)ceilf(totalDuration), noteBuf);
}

void Singer::SingPiece(TrackBuffer& buffer, const SingingPiece& piece, unsigned tempo, float RefFreq)
{
	std::vector<SingerNoteParams> noteParams;

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
				SingingPieceInternal _piece;
				_piece.lyric = lyric;
				_piece.notes = noteParams;
				NoteBuffer noteBuf;
				noteBuf.m_sampleRate = (float)buffer.Rate();
				noteBuf.m_cursorDelta = totalDuration;
				noteBuf.m_volume = m_noteVolume;

				GenerateWave(_piece, &noteBuf);
				buffer.WriteBlend(noteBuf);
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
		SingingPieceInternal _piece;
		_piece.lyric = lyric;
		_piece.notes = noteParams;
		NoteBuffer noteBuf;
		noteBuf.m_sampleRate = (float)buffer.Rate();
		noteBuf.m_cursorDelta = totalDuration;
		noteBuf.m_volume = m_noteVolume;

		GenerateWave(_piece, &noteBuf);
		buffer.WriteBlend(noteBuf);
	}

}

void Singer::RapAPiece(TrackBuffer& buffer, const RapPiece& piece, unsigned tempo, float RefFreq)
{
	float fduration = fabsf((float)(piece.m_duration * 60)) / (float)(tempo * 48);
	float fNumOfSamples = buffer.Rate()*fduration;

	if (piece.m_freq1<0.0 || piece.m_freq2<0.0)
	{
		if (piece.m_duration>0)
		{
			buffer.MoveCursor(fNumOfSamples);
			return;
		}
		else if (piece.m_duration<0)
		{
			buffer.MoveCursor(-fNumOfSamples);
			return;
		}
		else return;
	}

	RapPieceInternal _piece;
	_piece.lyric = piece.m_lyric;
	_piece.fNumOfSamples = fNumOfSamples;
	_piece.sampleFreq1 = RefFreq*piece.m_freq1 / (float)buffer.Rate();
	_piece.sampleFreq2 = RefFreq*piece.m_freq2 / (float)buffer.Rate();

	NoteBuffer noteBuf;
	noteBuf.m_sampleRate = (float)buffer.Rate();
	noteBuf.m_cursorDelta = fNumOfSamples;
	noteBuf.m_volume = m_noteVolume;

	GenerateWave_Rap(_piece, &noteBuf);

	buffer.WriteBlend(noteBuf);
}

void Singer::SingConsecutivePieces(TrackBuffer& buffer, const SingingSequence& pieces, unsigned tempo, float RefFreq)
{
	SingingPieceInternalList pieceList;

	float totalDuration = 0.0f;

	for (size_t j = 0; j < pieces.size(); j++)
	{
		const SingingPiece& piece = pieces[j];
		std::vector<SingerNoteParams> noteParams;

		for (size_t i = 0; i < piece.m_notes.size(); i++)
		{
			const Note& aNote = piece.m_notes[i];
			float fduration = fabsf((float)(aNote.m_duration * 60)) / (float)(tempo * 48);
			float fNumOfSamples = buffer.Rate()*fduration;
			if (aNote.m_freq_rel < 0.0f)
			{
				if (pieceList.size()>0 || noteParams.size()>0)
				{
					if (noteParams.size() > 0)
					{
						std::string lyric = piece.m_lyric;
						if (lyric == "") lyric = m_defaultLyric;
						SingingPieceInternal_Deferred _piece;
						_piece->lyric = lyric;
						_piece->notes = noteParams;
						pieceList.push_back(_piece);
					}
					NoteBuffer noteBuf;
					noteBuf.m_sampleRate = (float)buffer.Rate();
					noteBuf.m_cursorDelta = totalDuration;
					noteBuf.m_volume = m_noteVolume;

					GenerateWave_SingConsecutive(pieceList, &noteBuf);
					buffer.WriteBlend(noteBuf);
					noteParams.clear();
					pieceList.clear();
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
			SingingPieceInternal_Deferred _piece;
			_piece->lyric = lyric;
			_piece->notes = noteParams;
			pieceList.push_back(_piece);
		}		
	}

	if (pieceList.size() > 0)
	{
		NoteBuffer noteBuf;
		noteBuf.m_sampleRate = (float)buffer.Rate();
		noteBuf.m_cursorDelta = totalDuration;
		noteBuf.m_volume = m_noteVolume;

		GenerateWave_SingConsecutive(pieceList, &noteBuf);
		buffer.WriteBlend(noteBuf);
	}
}

void Singer::RapConsecutivePieces(TrackBuffer& buffer, const RapSequence& pieces, unsigned tempo, float RefFreq)
{
	RapPieceInternalList pieceList;

	float totalDuration = 0.0f;

	for (size_t j = 0; j < pieces.size(); j++)
	{
		const RapPiece& piece = pieces[j];
		float fduration = fabsf((float)(piece.m_duration * 60)) / (float)(tempo * 48);
		float fNumOfSamples = buffer.Rate()*fduration;

		if (piece.m_freq1 < 0.0f || piece.m_freq2 < 0.0f)
		{
			if (pieceList.size()>0)
			{
				NoteBuffer noteBuf;
				noteBuf.m_sampleRate = (float)buffer.Rate();
				noteBuf.m_cursorDelta = totalDuration;
				noteBuf.m_volume = m_noteVolume;

				GenerateWave_RapConsecutive(pieceList, &noteBuf);
				buffer.WriteBlend(noteBuf);
				pieceList.clear();
				totalDuration = 0.0f;
			}
			if (piece.m_duration>0)
			{
				buffer.MoveCursor(fNumOfSamples);
			}
			else if (piece.m_duration<0)
			{
				buffer.MoveCursor(-fNumOfSamples);
			}
		}
		else
		{

			RapPieceInternal_Deferred _piece;
			_piece->lyric = piece.m_lyric;
			_piece->fNumOfSamples = fNumOfSamples;
			_piece->sampleFreq1 = RefFreq*piece.m_freq1 / (float)buffer.Rate();
			_piece->sampleFreq2 = RefFreq*piece.m_freq2 / (float)buffer.Rate();

			totalDuration += fNumOfSamples;

			pieceList.push_back(_piece);

		}
	}
	if (pieceList.size() > 0)
	{
		NoteBuffer noteBuf;
		noteBuf.m_sampleRate = (float)buffer.Rate();
		noteBuf.m_cursorDelta = totalDuration;
		noteBuf.m_volume = m_noteVolume;

		GenerateWave_RapConsecutive(pieceList, &noteBuf);
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
		if(sscanf(cmd+7, "%f", &value))
			m_noteVolume = value;
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
