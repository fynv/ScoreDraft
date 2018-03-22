#include "SentenceGenerator_PSOLA.h"

void SentenceGenerator_PSOLA::GeneratePiece(bool isVowel, unsigned uSumLen, const float* freqMap, float& phase, Buffer& dstBuf, bool firstNote, bool hasNextNote, const SourceInfo& srcInfo, const SourceInfo& srcInfo_next, const SourceDerivedInfo& srcDerInfo)
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

	class SymmetricWindowWithPosition : public SymmetricWindow
	{
	public:
		float m_pos;
	};

	std::vector<SymmetricWindowWithPosition> windows;
	float fPeriodCount = 0.0f;

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

		unsigned winId = (unsigned)fPeriodCount;
		if (winId >= windows.size())
		{
			float srcHalfWinWidth = 1.0f / srcSampleFreq;
			Window srcWin;
			srcWin.CreateFromBuffer(srcInfo.source, (float)srcPos, srcHalfWinWidth);

			SymmetricWindowWithPosition symWin;
			symWin.CreateFromAsymmetricWindow(srcWin);
			symWin.m_pos = logicalPos;

			windows.push_back(symWin);

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

	std::vector<SymmetricWindowWithPosition> windows_next;

	if (hasNextNote)
	{
		float fPeriodCount = 0.0f;
		float logicalPos = 1.0f - srcDerInfo.preutter_pos_next*srcDerInfo.fixed_Weight;

		for (unsigned srcPos = 0; (float)srcPos <srcDerInfo.preutter_pos_next; srcPos++)
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

			unsigned winId = (unsigned)fPeriodCount;
			if (winId >= windows_next.size())
			{
				float srcHalfWinWidth = 1.0f / srcSampleFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(srcInfo_next.source, (float)srcPos, srcHalfWinWidth);

				SymmetricWindowWithPosition symWin;
				symWin.CreateFromAsymmetricWindow(srcWin);
				symWin.m_pos = logicalPos;

				windows_next.push_back(symWin);
			}
			fPeriodCount += srcSampleFreq;
			logicalPos += srcDerInfo.fixed_Weight;
		}
	}

	if (windows_next.size() == 0) hasNextNote = false;

	float tempHalfWinLen = 1.0f / minSampleFreq;

	unsigned winId0 = 0;
	unsigned winId0_next = 0;
	unsigned pos_final = 0;

	while (phase > -1.0f) phase -= 1.0f;

	float fTmpWinCenter;
	float transitionEnd = 1.0f - (srcDerInfo.preutter_pos_next - srcDerInfo.overlap_pos_next)*srcDerInfo.fixed_Weight;
	float transitionStart = transitionEnd* (1.0f - _transition);

	for (fTmpWinCenter = phase*tempHalfWinLen; fTmpWinCenter - tempHalfWinLen <= tempLen; fTmpWinCenter += tempHalfWinLen)
	{
		while (fTmpWinCenter > stretchingMap[pos_final] && pos_final<uSumLen - 1) pos_final++;
		float fWinPos = (float)pos_final / float(uSumLen);

		bool in_transition = hasNextNote && _transition > 0.0f && _transition < 1.0f && fWinPos >= transitionStart;

		unsigned winId1 = winId0 + 1;
		while (winId1 < windows.size() && windows[winId1].m_pos < fWinPos)
		{
			winId0++;
			winId1 = winId0 + 1;
		}
		if (winId1 == windows.size()) winId1 = winId0;

		unsigned winId1_next = winId0_next + 1;

		if (in_transition)
		{
			while (winId1_next < windows_next.size() && windows_next[winId1_next].m_pos < fWinPos)
			{
				winId0_next++;
				winId1_next = winId0_next + 1;
			}
			if (winId1_next == windows_next.size()) winId1_next = winId0_next;
		}

		SymmetricWindowWithPosition& win0 = windows[winId0];
		SymmetricWindowWithPosition& win1 = windows[winId1];

		float k;
		if (fWinPos >= win1.m_pos) k = 1.0f;
		else if (fWinPos <= win0.m_pos) k = 0.0f;
		else
		{
			k = (fWinPos - win0.m_pos) / (win1.m_pos - win0.m_pos);
		}

		float destSampleFreq;
		destSampleFreq = freqMap[pos_final];
		float destHalfWinLen = powf(2.0f, _gender) / destSampleFreq;

		SymmetricWindow shiftedWin0;
		SymmetricWindow shiftedWin1;

		SymmetricWindow l_win;
		SymmetricWindow* destWin = &l_win;

		shiftedWin0.Repitch_FormantPreserved(win0, destHalfWinLen);

		if (winId0 == winId1)
		{
			destWin = &shiftedWin0;
		}
		else
		{
			shiftedWin1.Repitch_FormantPreserved(win1, destHalfWinLen);
			l_win.Interpolate(shiftedWin0, shiftedWin1, k, destHalfWinLen);
		}

		SymmetricWindow *win_final_dest = destWin;
		SymmetricWindow l_win_transit;

		if (in_transition)
		{
			SymmetricWindowWithPosition& win0_next = windows_next[winId0_next];
			SymmetricWindowWithPosition& win1_next = windows_next[winId1_next];

			float k;
			if (fWinPos >= win1_next.m_pos) k = 1.0f;
			else if (fWinPos <= win0_next.m_pos) k = 0.0f;
			else
			{
				k = (fWinPos - win0_next.m_pos) / (win1_next.m_pos - win0_next.m_pos);
			}

			SymmetricWindow shiftedWin0_next;
			SymmetricWindow shiftedWin1_next;

			SymmetricWindow l_win_next;
			SymmetricWindow* destWin_next = &l_win_next;

			shiftedWin0_next.Repitch_FormantPreserved(win0_next, destHalfWinLen);

			if (winId0_next == winId1_next)
			{
				destWin_next = &shiftedWin0_next;
			}
			else
			{
				shiftedWin1_next.Repitch_FormantPreserved(win1_next, destHalfWinLen);
				l_win_next.Interpolate(shiftedWin0_next, shiftedWin1_next, k, destHalfWinLen);
			}

			float x = (fWinPos - transitionEnd) / (transitionEnd*_transition);
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

	// post processing

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