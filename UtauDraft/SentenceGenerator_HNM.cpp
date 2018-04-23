#include "SentenceGenerator_HNM.h"

void SentenceGenerator_HNM::GeneratePiece(bool _isVowel, unsigned uSumLen, const float* freqMap, float& phase, Buffer& dstBuf, bool firstNote, bool hasNextNote, const SourceInfo& srcInfo, const SourceInfo& srcInfo_next, const SourceDerivedInfo& srcDerInfo)
{
	float minSampleFreq;

	/// calculate finalBuffer->tmpBuffer map
	minSampleFreq = FLT_MAX;
	for (unsigned pos = 0; pos < uSumLen; pos++)
	{
		float sampleFreq = freqMap[pos];
		if (sampleFreq < minSampleFreq) minSampleFreq = sampleFreq;
	}

	float* stretchingMap;
	stretchingMap = new float[uSumLen];

	float pos_tmpBuf = 0.0f;
	for (unsigned pos = 0; pos < uSumLen; pos++)
	{
		float sampleFreq;
		sampleFreq = freqMap[pos];

		float speed = sampleFreq / minSampleFreq;
		pos_tmpBuf += speed;
		stretchingMap[pos] = pos_tmpBuf;
	}

	float tempLen = stretchingMap[uSumLen - 1];
	unsigned uTempLen = (unsigned)ceilf(tempLen);

	Buffer tempBuf;
	tempBuf.m_sampleRate = srcInfo.source.m_sampleRate;
	tempBuf.m_data.resize(uTempLen);
	tempBuf.SetZero();

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

	std::vector<ParameterSetWithPos> parameters;

	float fPeriodCount = 0.0f;
	unsigned lastmaxVoiced = 0;

	float fStartPos = firstNote ? srcDerInfo.overlap_pos : srcDerInfo.preutter_pos;
	float logicalPos = 0.0f;

	if (fStartPos < 0.0f)
	{
		logicalPos += (-fStartPos)*(firstNote ? srcDerInfo.headerWeight : srcDerInfo.fixed_Weight);
		fStartPos = 0.0f;
	}
	unsigned startPos = (unsigned)fStartPos;

	for (unsigned srcPos = startPos; srcPos < srcInfo.source.m_data.size(); srcPos++)
	{
		float srcSampleFreq;
		float srcFreqPos = (srcInfo.srcbegin + (float)srcPos) / (float)srcInfo.frq.m_window_interval;
		unsigned uSrcFreqPos = (unsigned)srcFreqPos;
		float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

		float freq1 = (float)srcInfo.frq[uSrcFreqPos].freq;
		if (freq1 <= 55.0f) freq1 = (float)srcInfo.frq.m_key_freq;

		float freq2 = (float)srcInfo.frq[uSrcFreqPos + 1].freq;
		if (freq2 <= 55.0f) freq2 = (float)srcInfo.frq.m_key_freq;

		float sampleFreq1 = freq1 / (float)srcInfo.source.m_sampleRate;
		float sampleFreq2 = freq2 / (float)srcInfo.source.m_sampleRate;

		srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

		unsigned paramId = (unsigned)fPeriodCount;
		if (paramId >= parameters.size())
		{
			bool isVowel = _isVowel && ((float)srcPos >= srcDerInfo.fixed_end || (float)srcPos < srcDerInfo.overlap_pos);
			unsigned maxVoiced = 0;

			if (!isVowel)
			{
				float halfWinlen = 3.0f / srcSampleFreq;
				Window capture;
				capture.CreateFromBuffer(srcInfo.source, (float)srcPos, halfWinlen);

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
			if (_isVowel && (float)srcPos >= srcDerInfo.preutter_pos && maxVoiced < lastmaxVoiced)
				maxVoiced = lastmaxVoiced;

			lastmaxVoiced = maxVoiced;

			ParameterSetWithPos paramSet;

			float srcHalfWinWidth = 1.0f / srcSampleFreq;
			Window srcWin;
			srcWin.CreateFromBuffer(srcInfo.source, (float)srcPos, srcHalfWinWidth);

			AmpSpectrum harmSpec;
			harmSpec.CreateFromWindow(srcWin);
			paramSet.NoiseSpectrum.Allocate(srcWin.m_halfWidth);

			if (!isVowel)
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
			paramSet.m_pos = logicalPos;
			parameters.push_back(paramSet);
		}
		fPeriodCount += srcSampleFreq;

		if (firstNote && (float)srcPos < srcDerInfo.preutter_pos)
		{
			logicalPos += srcDerInfo.headerWeight;
		}
		else if ((float)srcPos < srcDerInfo.fixed_end)
		{
			logicalPos += srcDerInfo.fixed_Weight;
		}
		else
		{
			logicalPos += srcDerInfo.vowel_Weight;
		}
	}

	std::vector<ParameterSetWithPos> parameters_next;

	if (hasNextNote)
	{
		float fPeriodCount = 0.0f;
		float logicalPos = 1.0f - srcDerInfo.preutter_pos_next*srcDerInfo.fixed_Weight;

		unsigned lastmaxVoiced = 0;
		for (unsigned srcPos = 0; (float)srcPos < srcDerInfo.preutter_pos_next; srcPos++)
		{
			float srcSampleFreq;
			float srcFreqPos = (srcInfo_next.srcbegin + (float)srcPos) / (float)srcInfo_next.frq.m_window_interval;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float freq1 = (float)srcInfo_next.frq[uSrcFreqPos].freq;
			if (freq1 <= 55.0f) freq1 = (float)srcInfo_next.frq.m_key_freq;

			float freq2 = (float)srcInfo_next.frq[uSrcFreqPos + 1].freq;
			if (freq2 <= 55.0f) freq2 = (float)srcInfo_next.frq.m_key_freq;

			float sampleFreq1 = freq1 / (float)srcInfo_next.source.m_sampleRate;
			float sampleFreq2 = freq2 / (float)srcInfo_next.source.m_sampleRate;

			srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

			unsigned paramId = (unsigned)fPeriodCount;
			if (paramId >= parameters_next.size())
			{
				bool isVowel = _isVowel && (float)srcPos < srcDerInfo.overlap_pos_next;
				unsigned maxVoiced = 0;

				if (!isVowel)
				{
					float halfWinlen = 3.0f / srcSampleFreq;
					Window capture;
					capture.CreateFromBuffer(srcInfo_next.source, (float)srcPos, halfWinlen);

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
				if (_isVowel && (float)srcPos >= srcDerInfo.preutter_pos_next && maxVoiced < lastmaxVoiced)
					maxVoiced = lastmaxVoiced;

				lastmaxVoiced = maxVoiced;

				ParameterSetWithPos paramSet;

				float srcHalfWinWidth = 1.0f / srcSampleFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(srcInfo_next.source, (float)srcPos, srcHalfWinWidth);

				AmpSpectrum harmSpec;
				harmSpec.CreateFromWindow(srcWin);
				paramSet.NoiseSpectrum.Allocate(srcWin.m_halfWidth);

				if (!isVowel)
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
				paramSet.m_pos = logicalPos;
				parameters_next.push_back(paramSet);
			}

			fPeriodCount += srcSampleFreq;
			logicalPos += srcDerInfo.fixed_Weight;
		}
	}

	if (parameters_next.size() == 0) hasNextNote = false;

	float tempHalfWinLen = 1.0f / minSampleFreq;

	unsigned paramId0 = 0;
	unsigned paramId0_next = 0;
	unsigned pos_final = 0;

	while (phase > -1.0f) phase -= 1.0f;

	float fTmpWinCenter;
	float transitionEnd = 1.0f - (srcDerInfo.preutter_pos_next - srcDerInfo.overlap_pos_next)*srcDerInfo.fixed_Weight;
	float transitionStart = transitionEnd* (1.0f - _transition);

	for (fTmpWinCenter = phase*tempHalfWinLen; fTmpWinCenter - tempHalfWinLen <= tempLen; fTmpWinCenter += tempHalfWinLen)
	{
		while (fTmpWinCenter > stretchingMap[pos_final] && pos_final < uSumLen - 1) pos_final++;
		float fParamPos = (float)pos_final / float(uSumLen);

		bool in_transition = hasNextNote && _transition > 0.0f && _transition < 1.0f && fParamPos >= transitionStart;

		unsigned paramId1 = paramId0 + 1;
		while (paramId1 < parameters.size() && parameters[paramId1].m_pos < fParamPos)
		{
			paramId0++;
			paramId1 = paramId0 + 1;
		}
		if (paramId1 == parameters.size()) paramId1 = paramId0;

		unsigned paramId1_next = paramId0_next + 1;

		if (in_transition)
		{
			while (paramId1_next < parameters_next.size() && parameters_next[paramId1_next].m_pos < fParamPos)
			{
				paramId0_next++;
				paramId1_next = paramId0_next + 1;
			}
			if (paramId1_next == parameters_next.size()) paramId1_next = paramId0_next;
		}

		ParameterSetWithPos& param0 = parameters[paramId0];
		ParameterSetWithPos& param1 = parameters[paramId1];

		float k;
		if (fParamPos >= param1.m_pos) k = 1.0f;
		else if (fParamPos <= param0.m_pos) k = 0.0f;
		else
		{
			k = (fParamPos - param0.m_pos) / (param1.m_pos - param0.m_pos);
		}

		float destSampleFreq;
		destSampleFreq = freqMap[pos_final];
		float destHalfWinLen = powf(2.0f, _gender) / destSampleFreq;

		ParameterSet scaledParam0;
		ParameterSet scaledParam1;

		ParameterSet l_param;
		ParameterSet* destParam = &l_param;

		scaledParam0.Scale(param0, destHalfWinLen);
		if (paramId0 == paramId1)
		{
			destParam = &scaledParam0;
		}
		else
		{
			scaledParam1.Scale(param1, destHalfWinLen);
			l_param.Interpolate(scaledParam0, scaledParam1, k);
		}

		ParameterSet* finalDestParam = destParam;
		ParameterSet l_paramTransit;

		if (in_transition)
		{
			ParameterSetWithPos& param0_next = parameters_next[paramId0_next];
			ParameterSetWithPos& param1_next = parameters_next[paramId1_next];

			float k;
			if (fParamPos >= param1_next.m_pos) k = 1.0f;
			else if (fParamPos <= param0_next.m_pos) k = 0.0f;
			else
			{
				k = (fParamPos - param0_next.m_pos) / (param1_next.m_pos - param0_next.m_pos);
			}

			ParameterSet scaledParam0_next;
			ParameterSet scaledParam1_next;

			ParameterSet l_param_next;
			ParameterSet* destParam_next = &l_param_next;

			scaledParam0_next.Scale(param0_next, destHalfWinLen);
			if (paramId0_next == paramId1_next)
			{
				destParam_next = &scaledParam0_next;
			}
			else
			{
				scaledParam1_next.Scale(param1_next, destHalfWinLen);
				l_param_next.Interpolate(scaledParam0_next, scaledParam1_next, k);
			}
			float k2;
			if (fParamPos >= transitionEnd)
				k2 = 1.0f;
			else
			{
				float x = (fParamPos - transitionEnd) / (transitionEnd*_transition);
				k2 = 0.5f*(cosf(x*(float)PI) + 1.0f);
			}
			finalDestParam = &l_paramTransit;

			l_paramTransit.Interpolate(*destParam, *destParam_next, k2);
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
		float pos_tmpBuf = stretchingMap[pos];
		float sampleFreq;
		sampleFreq = freqMap[pos];

		float speed = sampleFreq / minSampleFreq;

		int ipos1 = (int)ceilf(pos_tmpBuf - speed*0.5f);
		int ipos2 = (int)floorf(pos_tmpBuf + speed*0.5f);

		float sum = 0.0f;
		for (int ipos = ipos1; ipos <= ipos2; ipos++)
		{
			sum += tempBuf.GetSample(ipos);
		}
		float value = sum / (float)(ipos2 - ipos1 + 1);
		dstBuf.m_data[pos] = value;
	}

	delete[] stretchingMap;
}
