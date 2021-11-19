#include "SentenceDescriptor.h"
#include "SentenceGeneratorGeneral.h"

static float rate = 44100.0f;

inline void Clamp01(float& v)
{
	if (v < 0.0f) v = 0.0f;
	else if (v > 1.0f) v = 1.0f;
}

void RegulateSource(const float* srcData, unsigned len, Buffer& dstBuf, int srcStart, int srcEnd)
{
	unsigned uLen = (unsigned)(srcEnd - srcStart);
	dstBuf.Allocate(uLen);

	float acc = 0.0f;
	float count = 0.0f;
	for (unsigned i = 0; i < len; i++)
	{
		acc += srcData[i] * srcData[i];
		if (srcData[i] != 0.0f)
		{
			count += 1.0f;
		}
	}
	acc = sqrtf(count / acc)*0.3f;
	for (unsigned i = 0; i < uLen; i++)
	{
		int j = (int)i + srcStart;
		float v = 0.0f;
		if (j>=0 && j<len)
			v = srcData[j] * acc;
		dstBuf.m_data[i] = v;
	}
}

static void _floatBufSmooth(float* buf, unsigned size)
{
	static unsigned halfWinSize = 1024;
	static unsigned winSize = halfWinSize * 2;
	float *buf2 = new float[size];
	memset(buf2, 0, sizeof(float)*size);

	for (unsigned i = 0; i < size + halfWinSize; i += halfWinSize)
	{
		float sum = 0.0f;
		for (int j = -(int)halfWinSize; j < (int)halfWinSize; j++)
		{
			float v;
			int bufPos = (int)i + j;
			if (bufPos < 0) v = buf[0];
			else if (bufPos >= (int)size) v = buf[size - 1];
			else v = buf[bufPos];

			float x = (float)j / (float)halfWinSize*(float)PI;
			float w = (cosf(x) + 1.0f)*0.5f;

			sum += v*w;
		}
		float ave = sum / (float)halfWinSize;
		for (int j = -(int)halfWinSize; j < (int)halfWinSize; j++)
		{
			int bufPos = (int)i + j;
			if (bufPos < 0 || bufPos >= (int)size) continue;

			float x = (float)j / (float)halfWinSize*(float)PI;
			float w = (cosf(x) + 1.0f)*0.5f;

			buf2[bufPos] += w*ave;
		}
	}

	memcpy(buf, buf2, sizeof(float)*size);
	delete[] buf2;
}

void PreprocessFreqMap(const SentenceDescriptor* desc, unsigned outBufLen, float* freqMap, std::vector<unsigned>& bounds)
{
	unsigned i_piece = 0;
	unsigned i_freq = 0;

	bounds.clear();
	bounds.push_back(0);

	const std::vector<GeneralCtrlPnt>& piece_map = desc->piece_map;
	const std::vector<GeneralCtrlPnt>& freq_map = desc->freq_map;

	float lastPiece = piece_map[0].value;

	for (unsigned i = 0; i < outBufLen; i++)
	{
		float fpos = (float)i / rate*1000.0f;
		while (i_piece + 1<piece_map.size() && fpos >= piece_map[i_piece + 1].dstPos)
			i_piece++;
		float k_piece = (fpos - piece_map[i_piece].dstPos) / (piece_map[i_piece + 1].dstPos - piece_map[i_piece].dstPos);
		Clamp01(k_piece);
		float piece = piece_map[i_piece].value*(1.0f - k_piece) + piece_map[i_piece + 1].value*k_piece;

		if (i_freq + 1<freq_map.size() && fpos >= freq_map[i_freq + 1].dstPos)
		{
			if (piece - lastPiece >= 1.0f)
			{
				bounds.push_back(i);
				lastPiece = piece;
			}
			i_freq++;
		}

		while (i_freq + 1 < freq_map.size() && fpos >= freq_map[i_freq + 1].dstPos)
			i_freq++;
		float k_freq = (fpos - freq_map[i_freq].dstPos) / (freq_map[i_freq + 1].dstPos - freq_map[i_freq].dstPos);
		Clamp01(k_freq);
		float freq = freq_map[i_freq].value*(1.0f - k_freq) + freq_map[i_freq + 1].value*k_freq;

		freqMap[i] = freq / rate;
	}

	bounds.push_back(outBufLen);

	_floatBufSmooth(freqMap, outBufLen);

}
