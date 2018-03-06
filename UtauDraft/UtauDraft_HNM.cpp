#include "UtauDraft.h"

void UtauDraft::GenWaveStruct::_generateWave_HNM()
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
	tempBuf.m_sampleRate = source.m_sampleRate;
	tempBuf.m_data.resize(uTempLen);
	tempBuf.SetZero();

	class ParameterSet
	{
	public:
		AmpSpectrum HarmSpectrum;
		AmpSpectrum NoiseSpectrum;

		void Scale(const ParameterSet& src, float targetHalfWidth)
		{
			HarmSpectrum.Scale(src.HarmSpectrum, targetHalfWidth);
			NoiseSpectrum.Scale(src.NoiseSpectrum, targetHalfWidth);
		}

		void Interpolate(const ParameterSet& param0, const ParameterSet& param1, float k)
		{
			HarmSpectrum.Interpolate(param0.HarmSpectrum, param1.HarmSpectrum, k, param0.HarmSpectrum.m_halfWidth);
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
	float logicalPos = firstNote ? (-overlap_pos*headerWeight) : (-preutter_pos* fixed_Weight);

	unsigned lastmaxVoiced = 0;

	for (unsigned srcPos = 0; srcPos < source.m_data.size(); srcPos++)
	{
		float srcSampleFreq;
		float srcFreqPos = (srcbegin + (float)srcPos) / (float)frq.m_window_interval;
		unsigned uSrcFreqPos = (unsigned)srcFreqPos;
		float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

		float freq1 = (float)frq[uSrcFreqPos].freq;
		if (freq1 <= 55.0f) freq1 = (float)frq.m_key_freq;

		float freq2 = (float)frq[uSrcFreqPos + 1].freq;
		if (freq2 <= 55.0f) freq2 = (float)frq.m_key_freq;

		float sampleFreq1 = freq1 / (float)source.m_sampleRate;
		float sampleFreq2 = freq2 / (float)source.m_sampleRate;

		srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

		unsigned paramId = (unsigned)fPeriodCount;
		if (paramId >= parameters.size())
		{
			bool isVowel = _isVowel && ((float)srcPos >= fixed_end || (float)srcPos < overlap_pos);
			unsigned maxVoiced = 0;

			if (!isVowel)
			{
				float halfWinlen = 3.0f / srcSampleFreq;
				Window capture;
				capture.CreateFromBuffer(source, (float)srcPos, halfWinlen);

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
			if (_isVowel && (float)srcPos >= preutter_pos && maxVoiced < lastmaxVoiced)
				maxVoiced = lastmaxVoiced;

			lastmaxVoiced = maxVoiced;

			ParameterSetWithPos paramSet; 

			float srcHalfWinWidth = 1.0f / srcSampleFreq;
			Window srcWin;
			srcWin.CreateFromBuffer(source, (float)srcPos, srcHalfWinWidth);

			paramSet.HarmSpectrum.CreateFromWindow(srcWin);
			paramSet.NoiseSpectrum.Allocate(paramSet.HarmSpectrum.m_halfWidth);

			if (!isVowel)
			{
				for (unsigned i = maxVoiced + 1; i < (unsigned)paramSet.HarmSpectrum.m_data.size(); i++)
				{
					float amplitude = paramSet.HarmSpectrum.m_data[i];
					if (i < (unsigned)paramSet.NoiseSpectrum.m_data.size())
					{
						paramSet.NoiseSpectrum.m_data[i] = paramSet.HarmSpectrum.m_data[i];
						paramSet.HarmSpectrum.m_data[i] = 0.0f;
					}
				}
			}

			paramSet.m_pos = logicalPos;
			parameters.push_back(paramSet);
		}
		fPeriodCount += srcSampleFreq;

		if (firstNote && (float)srcPos < preutter_pos)
		{
			logicalPos += headerWeight;
		}
		else if ((float)srcPos < fixed_end)
		{
			logicalPos += fixed_Weight;
		}
		else
		{
			logicalPos += vowel_Weight;
		}
	}

	std::vector<ParameterSetWithPos> parameters_next;

	if (hasNextNote)
	{
		float fPeriodCount = 0.0f;
		float logicalPos = 1.0f - preutter_pos_next*fixed_Weight;

		unsigned lastmaxVoiced = 0;
		for (unsigned srcPos = 0; (float)srcPos < preutter_pos_next; srcPos++)
		{
			float srcSampleFreq;
			float srcFreqPos = (nextbegin + (float)srcPos) / (float)frq_next.m_window_interval;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float freq1 = (float)frq_next[uSrcFreqPos].freq;
			if (freq1 <= 55.0f) freq1 = (float)frq_next.m_key_freq;

			float freq2 = (float)frq_next[uSrcFreqPos + 1].freq;
			if (freq2 <= 55.0f) freq2 = (float)frq_next.m_key_freq;

			float sampleFreq1 = freq1 / (float)source_next.m_sampleRate;
			float sampleFreq2 = freq2 / (float)source_next.m_sampleRate;

			srcSampleFreq = sampleFreq1*(1.0f - fracSrcFreqPos) + sampleFreq2*fracSrcFreqPos;

			unsigned paramId = (unsigned)fPeriodCount;
			if (paramId >= parameters_next.size())
			{
				bool isVowel = _isVowel && (float)srcPos < overlap_pos_next;
				unsigned maxVoiced = 0;

				if (!isVowel)
				{
					float halfWinlen = 3.0f / srcSampleFreq;
					Window capture;
					capture.CreateFromBuffer(source_next, (float)srcPos, halfWinlen);

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
				if (_isVowel && (float)srcPos >= preutter_pos_next && maxVoiced < lastmaxVoiced)
					maxVoiced = lastmaxVoiced;

				lastmaxVoiced = maxVoiced;

				ParameterSetWithPos paramSet;

				float srcHalfWinWidth = 1.0f / srcSampleFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(source_next, (float)srcPos, srcHalfWinWidth);

				paramSet.HarmSpectrum.CreateFromWindow(srcWin);
				paramSet.NoiseSpectrum.Allocate(paramSet.HarmSpectrum.m_halfWidth);

				if (!isVowel)
				{
					for (unsigned i = maxVoiced + 1; i < (unsigned)paramSet.HarmSpectrum.m_data.size(); i++)
					{
						float amplitude = paramSet.HarmSpectrum.m_data[i];
						if (i < (unsigned)paramSet.NoiseSpectrum.m_data.size())
						{
							paramSet.NoiseSpectrum.m_data[i] = paramSet.HarmSpectrum.m_data[i];
							paramSet.HarmSpectrum.m_data[i] = 0.0f;
						}
					}
				}

				paramSet.m_pos = logicalPos;
				parameters_next.push_back(paramSet);
			}

			fPeriodCount += srcSampleFreq;
			logicalPos += fixed_Weight;
		}
	}

	if (parameters_next.size() == 0) hasNextNote = false;

	float tempHalfWinLen = 1.0f / minSampleFreq;

	unsigned paramId0 = 0;
	unsigned paramId0_next = 0;
	unsigned pos_final = 0;

	float& phase = *_phase;
	while (phase > -1.0f) phase -= 1.0f;

	float fTmpWinCenter;
	float transitionEnd = 1.0f - (preutter_pos_next - overlap_pos_next)*fixed_Weight;
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
			float x = (fParamPos - transitionEnd) / (transitionEnd*_transition);
			if (x > 0.0f) x = 0.0f;
			float k2 = 0.5f*(cosf(x*(float)PI) + 1.0f);
			finalDestParam = &l_paramTransit;

			l_paramTransit.Interpolate(*destParam, *destParam_next, k2);
		}

		if (finalDestParam->HarmSpectrum.NonZero())
		{
			SymmetricWindow destWin;
			destWin.CreateFromAmpSpec(finalDestParam->HarmSpectrum, tempHalfWinLen);
			destWin.MergeToBuffer(tempBuf, fTmpWinCenter);
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
