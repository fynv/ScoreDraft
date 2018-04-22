#include "SentenceGenerator_CPU.h"
#include "TrackBuffer.h"

void SentenceGenerator_CPU::_generatePiece(const UtauSourceFetcher& srcFetcher, const char* lyric, const char* lyric_next, unsigned uSumLen, const float* freqMap, NoteBuffer* noteBuf, unsigned noteBufPos, float& phase, bool firstNote, bool isVowel)
{
	SourceInfo srcInfo;
	SourceInfo srcInfo_next;
	SourceDerivedInfo srcDerInfo;

	bool hasNextNote = lyric_next != nullptr;

	if (!srcFetcher.FetchSourceInfo(lyric, srcInfo, !isVowel? _constVC : -1.0f)) return;
	if (hasNextNote && !srcFetcher.FetchSourceInfo(lyric_next, srcInfo_next)) return;
	srcDerInfo.DeriveInfo(firstNote, hasNextNote, uSumLen, srcInfo, srcInfo_next);

	Buffer dstBuf;
	dstBuf.m_sampleRate = srcInfo.source.m_sampleRate;
	dstBuf.m_data.resize(uSumLen);

	GeneratePiece(isVowel, uSumLen, freqMap, phase, dstBuf, firstNote, hasNextNote, srcInfo, srcInfo_next, srcDerInfo);

	memcpy(noteBuf->m_data + noteBufPos, &dstBuf.m_data[0], sizeof(float)*uSumLen);
	
}


void SentenceGenerator_CPU::GenerateSentence(const UtauSourceFetcher& srcFetcher, unsigned numPieces, const std::string* lyrics, const unsigned* isVowel, const unsigned* lengths, const float *freqAllMap, NoteBuffer* noteBuf)
{
	unsigned noteBufPos = 0;
	float phase = 0.0f;

	for (unsigned j = 0; j < numPieces; j++)
	{
		unsigned uSumLen = lengths[j];
		if (uSumLen == 0) continue;
		const float *freqMap = freqAllMap + noteBufPos;

		const char* lyric_next = nullptr;
		if (j < numPieces - 1)
		{
			lyric_next = lyrics[j + 1].data();
		}

		_generatePiece(srcFetcher, lyrics[j].data(), lyric_next, uSumLen, freqMap, noteBuf, noteBufPos, phase, j == 0, isVowel[j] != 0);

		noteBufPos += uSumLen;

	}
}

