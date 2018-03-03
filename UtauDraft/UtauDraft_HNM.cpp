#include "UtauDraft.h"


inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}

inline float randGauss(float sd)
{
	return sd*sqrtf(-2.0f*logf(rand01()))*cosf(rand01()*(float)PI);
}


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

#define NOISE_HALF_WINDOW_L 8
#define NOISE_HALF_WINDOW (1<<NOISE_HALF_WINDOW_L)

	class ParameterSet
	{
	public:
		float m_pos;
		SymmetricWindow m_win;
		float m_noiseAmps[NOISE_HALF_WINDOW + 1];
	};

	std::vector<ParameterSet> parameters;

	class Filter : public SymmetricWindow::FFTCallBack
	{
	public:
		bool haveNoise;
		unsigned max_voiced;
		float win1_len;
		ParameterSet *param;

		virtual void process(DComp* fftBuf, unsigned l)
		{
			if (haveNoise)
			{
				unsigned fftLen = 1 << l;
				SymmetricWindow_Axis win1;
				win1.Allocate(win1_len);

				float voicedEngerySum = 0.0f;
				float voicedCount = 0.0f;

				float rate1 = win1_len / (fftLen / 2);

				for (unsigned i = max_voiced + 1; i < fftLen / 2; i++)
				{
					float amplitude = (float)DCAbs(&fftBuf[i]);

					if ((float)i < win1_len)
						win1.m_data[i] = amplitude*rate1;

					fftBuf[i].Re = 0.0f;
					fftBuf[i].Im = 0.0f;
					fftBuf[fftLen - i].Re = 0.0f;
					fftBuf[fftLen - i].Im = 0.0f;
				}

				SymmetricWindow_Axis win2;
				win2.Scale(win1, (float)(NOISE_HALF_WINDOW));

				float rate = sqrtf((float)(NOISE_HALF_WINDOW) / win1_len);
				for (unsigned i = 0; i < NOISE_HALF_WINDOW; i++)
				{
					param->m_noiseAmps[i] = win2.m_data[i]*rate;
				}
				param->m_noiseAmps[NOISE_HALF_WINDOW] = 0.0f;
			}
			else
			{
				memset(param->m_noiseAmps, 0, sizeof(float)*NOISE_HALF_WINDOW);
			}
		}
	} filter;

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
			bool isVowel = (float)srcPos >= fixed_end || (float)srcPos<overlap_pos;
			unsigned maxVoiced = 0;

			if (!isVowel)
			{
				float halfWinlen = 3.0f / srcSampleFreq;
				Window capture;
				capture.CreateFromBuffer(source, (float)srcPos, halfWinlen);

				unsigned l;
				unsigned potHalfWinLen;
				calcPOT((unsigned)ceilf(halfWinlen), potHalfWinLen, l);

				Window Scaled;
				Scaled.Scale(capture, (float)potHalfWinLen);

				DComp* fftBuf = new DComp[potHalfWinLen];
				memset(fftBuf, 0, sizeof(DComp)*potHalfWinLen);

				for (unsigned i = 0; i < potHalfWinLen; i++)
				{
					float v = Scaled.GetSample((int)i) + Scaled.GetSample((int)i - (int)potHalfWinLen);
					fftBuf[i].Re = v;
				}
				fft(fftBuf, l);

				unsigned char voiced_cache[3] = { 0, 0, 0 };
				unsigned char cache_pos = 0;

				for (unsigned i = 3; i < potHalfWinLen / 2; i += 3)
				{
					double absv0 = DCAbs(&fftBuf[i]);
					double absv1 = DCAbs(&fftBuf[i - 1]);
					double absv2 = DCAbs(&fftBuf[i + 1]);

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

				delete[] fftBuf;
			}
			if ((float)srcPos >= preutter_pos && maxVoiced < lastmaxVoiced)
				maxVoiced = lastmaxVoiced;

			lastmaxVoiced = maxVoiced;

			ParameterSet paramSet;

			filter.haveNoise = !isVowel;
			filter.max_voiced = maxVoiced;
			filter.win1_len = 0.5f / srcSampleFreq / powf(2.0f, _gender);
			filter.param = &paramSet;

			float srcHalfWinWidth = 1.0f / srcSampleFreq;
			Window srcWin;
			srcWin.CreateFromBuffer(source, (float)srcPos, srcHalfWinWidth);

			unsigned l;
			unsigned fftLen;
			calcPOT((unsigned)ceilf(srcHalfWinWidth), fftLen, l);

			Window scaledWin;
			scaledWin.Scale(srcWin, (float)fftLen);

			SymmetricWindow symWin;
			symWin.CreateFromAsymmetricWindow(scaledWin, &filter);

			paramSet.m_win.Scale(symWin, srcHalfWinWidth);
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

	std::vector<ParameterSet> parameters_next;

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
				bool isVowel =(float)srcPos<overlap_pos_next;
				unsigned maxVoiced = 0;

				if (!isVowel)
				{
					float halfWinlen = 3.0f / srcSampleFreq;
					Window capture;
					capture.CreateFromBuffer(source_next, (float)srcPos, halfWinlen);

					unsigned l;
					unsigned potHalfWinLen;
					calcPOT((unsigned)ceilf(halfWinlen), potHalfWinLen, l);

					Window Scaled;
					Scaled.Scale(capture, (float)potHalfWinLen);

					DComp* fftBuf = new DComp[potHalfWinLen];
					memset(fftBuf, 0, sizeof(DComp)*potHalfWinLen);

					for (unsigned i = 0; i < potHalfWinLen; i++)
					{
						float v = Scaled.GetSample((int)i) + Scaled.GetSample((int)i - (int)potHalfWinLen);
						fftBuf[i].Re = v;
					}

					fft(fftBuf, l);

					unsigned char voiced_cache[3] = { 0, 0, 0 };
					unsigned char cache_pos = 0;

					for (unsigned i = 3; i < potHalfWinLen / 2; i += 3)
					{
						double absv0 = DCAbs(&fftBuf[i]);
						double absv1 = DCAbs(&fftBuf[i - 1]);
						double absv2 = DCAbs(&fftBuf[i + 1]);

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

					delete[] fftBuf;
				}
				if ((float)srcPos >= preutter_pos_next && maxVoiced < lastmaxVoiced)
					maxVoiced = lastmaxVoiced;

				lastmaxVoiced = maxVoiced;

				ParameterSet paramSet;

				filter.haveNoise = !isVowel;
				filter.max_voiced = maxVoiced;
				filter.win1_len = 0.5f / srcSampleFreq / powf(2.0f, _gender);
				filter.param = &paramSet;

				float srcHalfWinWidth = 1.0f / srcSampleFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(source_next, (float)srcPos, srcHalfWinWidth);

				unsigned l;
				unsigned fftLen;
				calcPOT((unsigned)ceilf(srcHalfWinWidth), fftLen, l);

				Window scaledWin;
				scaledWin.Scale(srcWin, (float)fftLen);

				SymmetricWindow symWin;
				symWin.CreateFromAsymmetricWindow(scaledWin, &filter);

				paramSet.m_win.Scale(symWin, srcHalfWinWidth);
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
		while (fTmpWinCenter > stretchingMap[pos_final] && pos_final<uSumLen - 1) pos_final++;
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

		ParameterSet& param0 = parameters[paramId0];
		ParameterSet& param1 = parameters[paramId1];

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

		SymmetricWindow shiftedWin0;
		SymmetricWindow shiftedWin1;

		SymmetricWindow l_win;
		SymmetricWindow* destWin = &l_win;

		shiftedWin0.Repitch_FormantPreserved(param0.m_win, destHalfWinLen);

		if (paramId0 == paramId1)
		{
			destWin = &shiftedWin0;
		}
		else
		{
			shiftedWin1.Repitch_FormantPreserved(param1.m_win, destHalfWinLen);
			l_win.Interpolate(shiftedWin0, shiftedWin1, k, destHalfWinLen);
		}
		SymmetricWindow *win_final_dest = destWin;
		SymmetricWindow l_win_transit;

		if (in_transition)
		{
			ParameterSet& param0_next = parameters_next[paramId0_next];
			ParameterSet& param1_next = parameters_next[paramId1_next];

			float k;
			if (fParamPos >= param1_next.m_pos) k = 1.0f;
			else if (fParamPos <= param0_next.m_pos) k = 0.0f;
			else
			{
				k = (fParamPos - param0_next.m_pos) / (param1_next.m_pos - param0_next.m_pos);
			}

			SymmetricWindow shiftedWin0_next;
			SymmetricWindow shiftedWin1_next;

			SymmetricWindow l_win_next;
			SymmetricWindow* destWin_next = &l_win_next;

			shiftedWin0_next.Repitch_FormantPreserved(param0_next.m_win, destHalfWinLen);

			if (paramId0_next == paramId1_next)
			{
				destWin_next = &shiftedWin0_next;
			}
			else
			{
				shiftedWin1_next.Repitch_FormantPreserved(param1_next.m_win, destHalfWinLen);
				l_win_next.Interpolate(shiftedWin0_next, shiftedWin1_next, k, destHalfWinLen);
			}
			float x = (fParamPos - transitionEnd) / (transitionEnd*_transition);
			if (x > 0.0f) x = 0.0f;
			float k2 = 0.5f*(cosf(x*(float)PI) + 1.0f);
			win_final_dest = &l_win_transit;

			l_win_transit.Interpolate(*destWin, *destWin_next, k2, destHalfWinLen);
		}

		SymmetricWindow l_win2;
		SymmetricWindow *winToMerge = &l_win2;

		if (destHalfWinLen == tempHalfWinLen)
		{
			winToMerge = win_final_dest;
		}
		else
		{
			l_win2.Scale(*win_final_dest, tempHalfWinLen);
		}

		winToMerge->MergeToBuffer(tempBuf, fTmpWinCenter);

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
	
	paramId0 = 0;
	paramId0_next = 0;

	for (float f_pos_final = 0.0f; f_pos_final - (float)NOISE_HALF_WINDOW < uSumLen; f_pos_final += (float)NOISE_HALF_WINDOW)
	{
		float fParamPos = f_pos_final / float(uSumLen);

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

		ParameterSet& param0 = parameters[paramId0];
		ParameterSet& param1 = parameters[paramId1];

		float k;
		if (fParamPos >= param1.m_pos) k = 1.0f;
		else if (fParamPos <= param0.m_pos) k = 0.0f;
		else
		{
			k = (fParamPos - param0.m_pos) / (param1.m_pos - param0.m_pos);
		}

		float l_noiseAmps[NOISE_HALF_WINDOW + 1];
		float* dest_noiseAmps = l_noiseAmps;

		if (paramId0 == paramId1)
		{
			dest_noiseAmps = param0.m_noiseAmps;
		}
		else
		{
			for (unsigned i = 0; i < NOISE_HALF_WINDOW + 1; i++)
				l_noiseAmps[i] = (1.0f - k)*param0.m_noiseAmps[i] + k* param1.m_noiseAmps[i];
		}

		float* final_dest_noiseAmps = dest_noiseAmps;
		float l_noiseAmps_transit[NOISE_HALF_WINDOW + 1];

		if (in_transition)
		{
			ParameterSet& param0_next = parameters_next[paramId0_next];
			ParameterSet& param1_next = parameters_next[paramId1_next];

			float k;
			if (fParamPos >= param1_next.m_pos) k = 1.0f;
			else if (fParamPos <= param0_next.m_pos) k = 0.0f;
			else
			{
				k = (fParamPos - param0_next.m_pos) / (param1_next.m_pos - param0_next.m_pos);
			}

			float l_noiseAmps_next[NOISE_HALF_WINDOW + 1];
			float* dest_noiseAmps_next = l_noiseAmps_next;

			if (paramId0_next == paramId1_next)
			{
				dest_noiseAmps_next = param0_next.m_noiseAmps;
			}
			else
			{
				for (unsigned i = 0; i < NOISE_HALF_WINDOW + 1; i++)
					l_noiseAmps_next[i] = (1.0f - k)*param0_next.m_noiseAmps[i] + k* param1_next.m_noiseAmps[i];
			}

			float x = (fParamPos - transitionEnd) / (transitionEnd*_transition);
			if (x > 0.0f) x = 0.0f;
			float k2 = 0.5f*(cosf(x*(float)PI) + 1.0f);
			final_dest_noiseAmps = l_noiseAmps_transit;

			for (unsigned i = 0; i < NOISE_HALF_WINDOW + 1; i++)
				l_noiseAmps_transit[i] = (1.0f - k2)*dest_noiseAmps[i] + k2* dest_noiseAmps_next[i];
		}

		bool haveNoise = false;
		for (unsigned i = 1; i < NOISE_HALF_WINDOW; i++)
		{
			if (final_dest_noiseAmps[i] > 0.0f)
			{
				haveNoise = true;
				break;
			}
		}

		if (haveNoise)
		{

			DComp* fftBuf = new DComp[NOISE_HALF_WINDOW * 2];

			fftBuf[0].Re = 0.0;
			fftBuf[0].Im = 0.0;

			for (unsigned i = 1; i < NOISE_HALF_WINDOW; i++)
			{
				double angle = (double)rand01()*PI*2.0;
				DCSetAA(&fftBuf[i], 1.0, angle);
				fftBuf[i].Re *= final_dest_noiseAmps[i];
				fftBuf[i].Im *= final_dest_noiseAmps[i];

				fftBuf[NOISE_HALF_WINDOW * 2 - i].Re = fftBuf[i].Re;
				fftBuf[NOISE_HALF_WINDOW * 2 - i].Im = fftBuf[i].Im;
			}
			fftBuf[NOISE_HALF_WINDOW].Re = 0.0f;
			fftBuf[NOISE_HALF_WINDOW].Im = 0.0f;

			ifft(fftBuf, NOISE_HALF_WINDOW_L + 1);

			Window winToMerge;
			winToMerge.Allocate((float)NOISE_HALF_WINDOW);
			for (unsigned i = 0; i < NOISE_HALF_WINDOW; i++)
			{
				float window = (cosf((float)i * (float)PI / ((float)NOISE_HALF_WINDOW)) + 1.0f)*0.5f;
				winToMerge.SetSample((int)i, (float)fftBuf[i].Re*window);
				if (i > 0)
					winToMerge.SetSample(-(int)i, (float)fftBuf[NOISE_HALF_WINDOW * 2 - i].Re*window);
			}
			winToMerge.MergeToBuffer(dstBuf, f_pos_final);

			delete[] fftBuf;
		}

	}
}

