#include "SentenceDescriptor.h"
#include "SentenceGeneratorGeneral.h"
#include "SentenceGeneratorCPU.h"

#include "fft.h"
#include "VoiceUtil.h"
using namespace VoiceUtil;

static float rate = 44100.0f;

inline void Clamp01(float& v)
{
	if (v < 0.0f) v = 0.0f;
	else if (v > 1.0f) v = 1.0f;
}

void GenerateSentenceCPU(const SentenceDescriptor* desc, float* outBuf, unsigned outBufLen)
{
	class ParameterSet
	{
	public:
		SymmetricWindow HarmWindow;
		AmpSpectrum NoiseSpectrum;

		virtual ~ParameterSet(){}

		void Scale(const ParameterSet& src, float targetHalfWidth)
		{
			HarmWindow.Repitch_FormantPreserved(src.HarmWindow, targetHalfWidth);
			NoiseSpectrum.Scale(src.NoiseSpectrum, targetHalfWidth);
		}

		void Interpolate(const ParameterSet& param0, const ParameterSet& param1, float k)
		{
			HarmWindow.Interpolate(param0.HarmWindow, param1.HarmWindow, k, param0.HarmWindow.m_halfWidth);
			NoiseSpectrum.Interpolate(param0.NoiseSpectrum, param1.NoiseSpectrum, k, param0.NoiseSpectrum.m_halfWidth);
		}
	};

	class ParameterSetWithPos : public ParameterSet
	{
	public:
		float m_pos;
	};

	typedef std::vector<ParameterSetWithPos> ParameterVec;
	std::vector<ParameterVec> ParameterVecs;

	const std::vector<Piece>& pieces = desc->pieces;

	ParameterVecs.resize(pieces.size());

	for (size_t i = 0; i < pieces.size(); i++)
	{
		const Piece& piece = pieces[i];
		ParameterVec& parameters = ParameterVecs[i];

		int srcStart = (int)(piece.srcMap[0].srcPos*0.001f*rate);
		int srcEnd = (int)ceilf(piece.srcMap[piece.srcMap.size() - 1].srcPos*0.001f*rate);

		int bound = piece.src.frq.data.size() * piece.src.frq.interval;

		if (srcStart < 0)
		{
			srcStart = 0;
		}

		if (srcEnd > bound)
		{
			srcEnd = bound;
		}

		float fPeriodCount = 0.0f;
		unsigned i_srcMap = 0;
		unsigned lastmaxVoiced = 0;

		Buffer SrcBuffer;
		SrcBuffer.m_sampleRate = (unsigned)rate;
		RegulateSource(piece.src.wav.buf, piece.src.wav.len, SrcBuffer,srcStart,srcEnd);

		for (int srcPos = srcStart; srcPos < srcEnd; srcPos++)
		{
			float fsrcPos = (float)srcPos / rate*1000.0f;
			while (i_srcMap + 1 < piece.srcMap.size() && fsrcPos >= piece.srcMap[i_srcMap + 1].srcPos)
				i_srcMap++;

			int isVowel = piece.srcMap[i_srcMap].isVowel;

			float k_srcMap = (fsrcPos - piece.srcMap[i_srcMap].srcPos) / (piece.srcMap[i_srcMap + 1].srcPos - piece.srcMap[i_srcMap].srcPos);
			Clamp01(k_srcMap);
			float fdstPos = piece.srcMap[i_srcMap].dstPos*(1.0f - k_srcMap) + piece.srcMap[i_srcMap + 1].dstPos*k_srcMap;
			float dstPos = fdstPos*0.001f*rate;

			float srcSampleFreq;
			float srcFreqPos = (float)srcPos / (float)piece.src.frq.interval;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float freq1 = (float)piece.src.frq.data[uSrcFreqPos].freq;
			if (freq1 <= 55.0f) freq1 = (float)piece.src.frq.key;

			float freq2 = (float)piece.src.frq.data[uSrcFreqPos + 1].freq;
			if (freq2 <= 55.0f) freq2 = (float)piece.src.frq.key;

			float sampleFreq1 = freq1 / rate;
			float sampleFreq2 = freq2 / rate;

			srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

			unsigned paramId = (unsigned)fPeriodCount;
			if (paramId >= parameters.size())
			{
				unsigned maxVoiced = 0;
				if (isVowel < 2)
				{
					float halfWinlen = 3.0f / srcSampleFreq;
					Window capture;
					capture.CreateFromBuffer(SrcBuffer, (float)(srcPos - srcStart), halfWinlen);

					AmpSpectrum capSpec;
					capSpec.CreateFromWindow(capture);

					unsigned char voiced_cache[3] = { 0, 0, 0 };
					unsigned char cache_pos = 0;

					for (unsigned i = 3; i + 1 < capSpec.m_data.size(); i += 3)
					{
						double absv0 = capSpec.m_data[i];
						double absv1 = capSpec.m_data[i - 1];
						double absv2 = capSpec.m_data[i + 1];

						double rate = absv0 / (absv0 + absv1 + absv2);

						if (rate > 0.7)
						{
							voiced_cache[cache_pos] = 1;
						}
						else
						{
							voiced_cache[cache_pos] = 0;
						}

						cache_pos = (cache_pos + 1) % 3;

						if (voiced_cache[0] + voiced_cache[1] + voiced_cache[2] > 1)
						{
							maxVoiced = i / 3;
						}
					}
				}
				if (isVowel > 0 && maxVoiced < lastmaxVoiced)
					maxVoiced = lastmaxVoiced;

				lastmaxVoiced = maxVoiced;

				ParameterSetWithPos paramSet;

				float srcHalfWinWidth = 1.0f / srcSampleFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(SrcBuffer, (float)(srcPos - srcStart), srcHalfWinWidth);

				AmpSpectrum harmSpec;
				harmSpec.CreateFromWindow(srcWin);
				paramSet.NoiseSpectrum.Allocate(srcWin.m_halfWidth);

				if (isVowel < 2)
				{
					for (unsigned i = maxVoiced + 1; i < (unsigned)harmSpec.m_data.size(); i++)
					{
						float amplitude = harmSpec.m_data[i];
						harmSpec.m_data[i] = 0.0f;
						if (i < (unsigned)paramSet.NoiseSpectrum.m_data.size())
						{
							paramSet.NoiseSpectrum.m_data[i] = amplitude;
						}
					}
				}
				paramSet.HarmWindow.CreateFromAmpSpec(harmSpec);
				paramSet.m_pos = dstPos;
				parameters.push_back(paramSet);
			}

			fPeriodCount += srcSampleFreq;
		}
	}

	float* freqMap = new float[outBufLen];
	std::vector<unsigned> bounds;

	PreprocessFreqMap(desc, outBufLen, freqMap, bounds);

	float phase = 0.0f;
	unsigned i_pieceMap = 0;
	unsigned i_volumeMap = 0;

	const std::vector<GeneralCtrlPnt>& piece_map = desc->piece_map;
	const std::vector<GeneralCtrlPnt>& volume_map = desc->volume_map;

	for (unsigned i = 0; i < (unsigned)bounds.size()-1; i++)
	{
		float* pFreqMap = freqMap + bounds[i];
		unsigned uSumLen = bounds[i + 1] - bounds[i];

		float minSampleFreq = FLT_MAX;
		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float sampleFreq = pFreqMap[pos];
			if (sampleFreq < minSampleFreq) minSampleFreq = sampleFreq;
		}

		float* stretchingMap;
		stretchingMap = new float[uSumLen];

		float pos_tmpBuf = 0.0f;
		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			float sampleFreq;
			sampleFreq = pFreqMap[pos];

			float speed = sampleFreq / minSampleFreq;
			pos_tmpBuf += speed;
			stretchingMap[pos] = pos_tmpBuf;
		}

		float tempLen = stretchingMap[uSumLen - 1];
		unsigned uTempLen = (unsigned)ceilf(tempLen);

		Buffer tempBuf;
		tempBuf.m_sampleRate = (unsigned)rate;
		tempBuf.Allocate(uTempLen);

		float tempHalfWinLen = 1.0f / minSampleFreq;
		unsigned pos_local = 0;

		while (phase > -1.0f) phase -= 1.0f;

		float fTmpWinCenter;
		for (fTmpWinCenter = phase*tempHalfWinLen; fTmpWinCenter - tempHalfWinLen <= tempLen; fTmpWinCenter += tempHalfWinLen)
		{
			while (fTmpWinCenter > stretchingMap[pos_local] && pos_local < uSumLen - 1) pos_local++;
			unsigned pos_global = pos_local + bounds[i];
			float f_pos_global = (float)pos_global / rate*1000.0f;

			while (i_pieceMap + 1 < piece_map.size() && f_pos_global >= piece_map[i_pieceMap + 1].dstPos)
				i_pieceMap++;

			float k_piece = (f_pos_global - piece_map[i_pieceMap].dstPos) / (piece_map[i_pieceMap + 1].dstPos - piece_map[i_pieceMap].dstPos);
			Clamp01(k_piece);
			float fPieceId = piece_map[i_pieceMap].value* (1.0f - k_piece) + piece_map[i_pieceMap + 1].value*k_piece;
			
			unsigned pieceId0 = (unsigned)fPieceId;
			float pieceId_frac = fPieceId - (float)pieceId0;
			unsigned pieceId1 = (unsigned)fPieceId +1;
			if (pieceId0 >= (unsigned)pieces.size()) pieceId0 = (unsigned)pieces.size() - 1;
			if (pieceId_frac == 0.0f || pieceId1 >= (unsigned)pieces.size())
			{
				pieceId1 = pieceId0;
				pieceId_frac = 0.0f;
			}

			ParameterVec& parameters0 = ParameterVecs[pieceId0];
			ParameterVec& parameters1 = ParameterVecs[pieceId1];

			unsigned paramId00 = 0;
			unsigned paramId01 = 1;
			unsigned paramId10 = 0;
			unsigned paramId11 = 1;

			while (paramId01<parameters0.size() && parameters0[paramId01].m_pos <(float)pos_global)
			{
				paramId00++;
				paramId01 = paramId00 + 1;
			}
			if (paramId01 == parameters0.size()) paramId01 = paramId00;

			if (pieceId1 > pieceId0)
			{
				while (paramId11<parameters1.size() && parameters1[paramId11].m_pos <(float)pos_global)
				{
					paramId10++;
					paramId11 = paramId10 + 1;
				}
				if (paramId11 == parameters1.size()) paramId11 = paramId10;
			}

			ParameterSetWithPos& param00 = parameters0[paramId00];
			ParameterSetWithPos& param01 = parameters0[paramId01];

			float k0;
			if ((float)pos_global >= param01.m_pos) k0 = 1.0f;
			else if ((float)pos_global <= param00.m_pos) k0 = 0.0f;
			else
			{
				k0 = ((float)pos_global - param00.m_pos) / (param01.m_pos - param00.m_pos);
			}

			float destSampleFreq = pFreqMap[pos_local];
			float destHalfWinLen = 1.0f / destSampleFreq;

			ParameterSet scaledParam00;
			ParameterSet scaledParam01;

			ParameterSet l_param0;
			ParameterSet* destParam0 = &l_param0;

			scaledParam00.Scale(param00, destHalfWinLen);
			if (paramId00 == paramId01)
			{
				destParam0 = &scaledParam00;
			}
			else
			{
				scaledParam01.Scale(param01, destHalfWinLen);
				l_param0.Interpolate(scaledParam00, scaledParam01, k0);
			}

			ParameterSet* finalDestParam = destParam0;
			ParameterSet l_paramTransit;

			if (pieceId1 > pieceId0)
			{
				ParameterSetWithPos& param10 = parameters1[paramId10];
				ParameterSetWithPos& param11 = parameters1[paramId11];

				float k1;
				if ((float)pos_global >= param11.m_pos) k1 = 1.0f;
				else if ((float)pos_global <= param10.m_pos) k1 = 0.0f;
				else
				{
					k1 = ((float)pos_global - param10.m_pos) / (param11.m_pos - param10.m_pos);
				}

				ParameterSet scaledParam10;
				ParameterSet scaledParam11;

				ParameterSet l_param1;
				ParameterSet* destParam1 = &l_param1;

				scaledParam10.Scale(param10, destHalfWinLen);
				if (paramId10 == paramId11)
				{
					destParam1 = &scaledParam10;
				}
				else
				{
					scaledParam11.Scale(param11, destHalfWinLen);
					l_param1.Interpolate(scaledParam10, scaledParam11, k1);
				}
				finalDestParam = &l_paramTransit;
				l_paramTransit.Interpolate(*destParam0, *destParam1, pieceId_frac);
			}	

			if (finalDestParam->HarmWindow.NonZero())
			{
				SymmetricWindow l_destWin;
				SymmetricWindow *destWin = &l_destWin;
				if (finalDestParam->HarmWindow.m_halfWidth == tempHalfWinLen)
					destWin = &finalDestParam->HarmWindow;
				else
					l_destWin.Scale(finalDestParam->HarmWindow, tempHalfWinLen);
				destWin->MergeToBuffer(tempBuf, fTmpWinCenter);
			}

			if (finalDestParam->NoiseSpectrum.NonZero())
			{
				Window destWin;
				destWin.CreateFromAmpSpec_noise(finalDestParam->NoiseSpectrum, tempHalfWinLen);
				destWin.MergeToBuffer(tempBuf, fTmpWinCenter);
			}	

		}
		phase = (fTmpWinCenter - tempLen) / tempHalfWinLen;

		for (unsigned pos = 0; pos < uSumLen; pos++)
		{
			unsigned pos_global = pos + bounds[i];
			float f_pos_global = (float)pos_global / rate*1000.0f;
			while (i_volumeMap + 1 < volume_map.size() && f_pos_global >= volume_map[i_volumeMap + 1].dstPos)
				i_volumeMap++;
			float k_volume = (f_pos_global - volume_map[i_volumeMap].dstPos) / (volume_map[i_volumeMap + 1].dstPos - volume_map[i_volumeMap].dstPos);
			Clamp01(k_volume);
			float volume = volume_map[i_volumeMap].value* (1.0f - k_volume) + volume_map[i_volumeMap + 1].value*k_volume;

			float pos_tmpBuf = stretchingMap[pos];
			float sampleFreq;
			sampleFreq = pFreqMap[pos];

			float speed = sampleFreq / minSampleFreq;

			int ipos1 = (int)ceilf(pos_tmpBuf - speed*0.5f);
			int ipos2 = (int)floorf(pos_tmpBuf + speed*0.5f);

			if (ipos1 >= (int)uTempLen) ipos1 = (int)uTempLen - 1;
			if (ipos2 >= (int)uTempLen) ipos2 = (int)uTempLen - 1;

			float sum = 0.0f;
			for (int ipos = ipos1; ipos <= ipos2; ipos++)
			{
				sum += tempBuf.GetSample(ipos);
			}
			float value = sum / (float)(ipos2 - ipos1 + 1);
			outBuf[pos + bounds[i]] = value*volume;
		}

		delete[] stretchingMap;
	}	

	delete[] freqMap;

}
