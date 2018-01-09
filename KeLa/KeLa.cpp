#include "PyScoreDraft.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include <string.h>
#include <cmath>
#include <ReadWav.h>
#include "FrequencyDetection.h"
#include <float.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include "fft.h"

struct Buffer
{
	unsigned m_sampleRate;
	std::vector<float> m_data;
	float GetSample(int i) const
	{
		size_t usize = m_data.size();
		if (i<0 || i >= (int)usize) return 0.0f;
		return m_data[i];
	}

	float GetMax()
	{
		float maxv = 0.0f;
		for (size_t i = 0; i < m_data.size(); i++)
		{
			if (fabsf(m_data[i])>maxv) maxv = fabsf(m_data[i]);
		}
		return maxv;
	}

	void SetZero()
	{
		memset(m_data.data(), 0, sizeof(float)*m_data.size());
	}
};

bool ReadWavToBuffer(const char* filename, Buffer& buf, float& maxV)
{
	ReadWav reader;
	if (!reader.OpenFile(filename)) return false;
	unsigned numSamples;
	if (!reader.ReadHeader(buf.m_sampleRate, numSamples)) return false;
	buf.m_data.resize((size_t)numSamples);
	return reader.ReadSamples(buf.m_data.data(), numSamples, maxV);
}

class Window
{
public:
	float m_halfWidth;
	std::vector<float> m_data;

	virtual float GetSample(int i) const
	{
		size_t u_width = m_data.size();
		if (i <= -(int)u_width || i >= (int)u_width) return 0.0f;
		unsigned pos = (unsigned)(i + (int)u_width) % u_width;
		return m_data[pos];
	}

	virtual void SetSample(int i, float v)
	{
		size_t u_width = m_data.size();
		if (i <= -(int)u_width || i >= (int)u_width) return;
		unsigned pos = (unsigned)(i + (int)u_width) % u_width;
		m_data[pos] = v;
	}

	void SetZero()
	{
		memset(m_data.data(), 0, sizeof(float)*m_data.size());
	}

	void CreateFromBuffer(const Buffer& src, float center, float halfWidth)
	{
		unsigned u_halfWidth = (unsigned)ceilf(halfWidth);
		unsigned u_width = u_halfWidth << 1;

		m_halfWidth = halfWidth;
		m_data.resize(u_width);

		unsigned u_Center = (unsigned)center;

		for (int i = -(int)u_halfWidth; i < (int)u_halfWidth; i++)
		{
			float window = (cosf((float)i * (float)PI / halfWidth) + 1.0f)*0.5f;

			int srcIndex = (int)u_Center + i;
			float v_src = src.GetSample(srcIndex);

			SetSample(i, window* v_src);
		}
	}

	void MergeToBuffer(Buffer& buf, float pos)
	{
		int ipos = (int)pos;
		unsigned u_halfWidth = (unsigned)ceilf(m_halfWidth);
		unsigned u_width = u_halfWidth << 1;

		for (int i = max(-(int)u_halfWidth, -ipos); i < (int)u_halfWidth; i++)
		{
			int dstIndex = ipos + i;
			if (dstIndex >= (int)buf.m_data.size()) break;
			buf.m_data[dstIndex] += GetSample(i);
		}
	}

};


class SymmetricWindow : public Window
{
public:
	virtual float GetSample(int i) const
	{
		if (i < 0) i = -i;
		if (i >= m_data.size()) return 0.0f;
		return m_data[i];
	}
	virtual void SetSample(int i, float v)
	{
		if (i < 0) i = -i;
		if (i >= m_data.size()) return;
		m_data[i] = v;
	}

	void CreateFromAsymmetricWindow(const Window& src)
	{
		unsigned u_srcHalfWidth = (unsigned)ceilf(src.m_halfWidth);
		unsigned u_srcWidth = u_srcHalfWidth << 1;

		unsigned l = 0;
		unsigned fftLen = 1;
		while (fftLen < u_srcWidth)
		{
			l++;
			fftLen <<= 1;
		}

		m_halfWidth = src.m_halfWidth;
		m_data.resize(fftLen / 2);

		DComp* fftBuf = new DComp[fftLen];
		memset(fftBuf, 0, sizeof(DComp)*fftLen);

		for (unsigned i = 0; i < u_srcHalfWidth; i++)
		{
			fftBuf[i].Re = (double)src.m_data[i];
			fftBuf[fftLen - 1 - i].Re = (double)src.m_data[u_srcWidth - 1 - i];
		}

		fft(fftBuf, l);

		fftBuf[0].Re = 0.0f;
		fftBuf[0].Im = 0.0f;

		for (unsigned i = 1; i < fftLen; i++)
		{
			double absv = DCAbs(&fftBuf[i]);
			fftBuf[i].Re = absv;
			fftBuf[i].Im = 0.0f;
		}

		ifft(fftBuf, l);

		for (unsigned i = 0; i < fftLen / 2; i++)
			m_data[i] = (float)fftBuf[i].Re;

		delete[] fftBuf;

	}

	void Scale(const SymmetricWindow& src, float targetHalfWidth)
	{
		m_halfWidth = targetHalfWidth;
		unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
		m_data.resize(u_TargetHalfWidth);

		float rate = src.m_halfWidth / targetHalfWidth;
		bool interpolation = rate < 1.0f;
		for (unsigned i = 0; i < u_TargetHalfWidth; i++)
		{
			float destValue;
			float srcPos = (float)i*rate;
			if (interpolation)
			{
				int ipos1 = (int)floorf(srcPos);
				float frac = srcPos - (float)ipos1;
				int ipos2 = ipos1 + 1;
				int ipos0 = ipos1 - 1;
				int ipos3 = ipos1 + 2;

				float p0 = src.GetSample(ipos0);
				float p1 = src.GetSample(ipos1);
				float p2 = src.GetSample(ipos2);
				float p3 = src.GetSample(ipos3);

				destValue = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceilf(srcPos - rate*0.5f);
				int ipos2 = (int)floorf(srcPos + rate*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += src.GetSample(ipos);
				}
				destValue = sum / (float)(ipos2 - ipos1 + 1);

			}

			m_data[i] = destValue;

		}
	}

	void Repitch_FormantPreserved(const SymmetricWindow& src, float targetHalfWidth)
	{
		m_halfWidth = targetHalfWidth;
		unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
		m_data.resize(u_TargetHalfWidth);

		float srcHalfWidth = src.m_halfWidth;
		unsigned uSrcHalfWidth = (unsigned)src.m_data.size();
		float rate = targetHalfWidth / srcHalfWidth;

		float targetWidth = targetHalfWidth*2.0f;

		for (unsigned i = 0; (float)i < targetHalfWidth; i++)
		{
			m_data[i] = 0.0f;

			float srcPos = (float)i;
			unsigned uSrcPos = (unsigned)(srcPos + 0.5f);

			while (uSrcPos < uSrcHalfWidth)
			{
				m_data[i] += src.m_data[uSrcPos];
				srcPos += targetWidth;
				uSrcPos = (unsigned)(srcPos + 0.5f);
			}

			srcPos = targetWidth - (float)i;
			uSrcPos = (unsigned)(srcPos + 0.5f);

			while (uSrcPos < uSrcHalfWidth)
			{
				m_data[i] += src.m_data[uSrcPos];
				srcPos += targetWidth;
				uSrcPos = (unsigned)(srcPos + 0.5f);
			}

			srcPos = (float)i + targetHalfWidth;
			uSrcPos = (unsigned)(srcPos + 0.5f);

			while (uSrcPos < uSrcHalfWidth)
			{
				m_data[i] += src.m_data[uSrcPos];
				srcPos += targetWidth;
				uSrcPos = (unsigned)(srcPos + 0.5f);
			}

			srcPos = -(float)i + targetHalfWidth;
			uSrcPos = (unsigned)(srcPos + 0.5f);

			while (uSrcPos < uSrcHalfWidth)
			{
				m_data[i] += src.m_data[uSrcPos];
				srcPos += targetWidth;
				uSrcPos = (unsigned)(srcPos + 0.5f);
			}
		}

		// rewindow
		float amplitude = sqrtf(rate);
		for (unsigned i = 0; (float)i < targetHalfWidth; i++)
		{
			float window = (cosf((float)i * (float)PI / targetHalfWidth) + 1.0f)*0.5f;
			m_data[i] *= amplitude*window;
		}


	}

};

struct SymmetricWindowWithPosition
{
	SymmetricWindow win;
	float center;
};


void DetectFreqs(const Buffer& buf, std::vector<float>& frequencies, unsigned step, float& ave_freq)
{
	unsigned halfWinLen = 1024;
	float* temp = new float[halfWinLen * 2];

	for (unsigned center = 0; center < buf.m_data.size(); center += step)
	{
		Window win;
		win.CreateFromBuffer(buf, (float)center, (float)halfWinLen);

		for (int i = -(int)halfWinLen; i < (int)halfWinLen; i++)
			temp[i + halfWinLen] = win.GetSample(i);

		float freq = fetchFrequency(halfWinLen * 2, temp, buf.m_sampleRate);

		frequencies.push_back(freq);
	}

	delete[] temp;

	struct Range
	{
		unsigned begin;
		unsigned end;

		Range()
		{
			begin = (unsigned)(-1);
			end = (unsigned)(-1);
		}
		unsigned Length()
		{
			return end - begin;
		}
	};

	Range BestRange;
	Range CurRange;

	for (size_t i = 0; i < frequencies.size(); i++)
	{
		if (frequencies[i] > 0)
		{
			if (CurRange.begin == (unsigned)(-1))
			{
				CurRange.begin = (unsigned)i;
			}
			else
			{
				CurRange.end = (unsigned)i;
				if (CurRange.Length() > BestRange.Length()) BestRange = CurRange;
			}
		}
		else
		{
			CurRange.begin = (unsigned)(-1);
		}
	}

	ave_freq = 0.0f;
	float count = 0.0f;
	unsigned sampleBegin = (unsigned) (-1);
	size_t sampleEnd = (unsigned)(-1);
	for (size_t i = 0; i < frequencies.size(); i++)
	{
		if ((i<BestRange.begin || i>BestRange.end) && frequencies[i] > 0.0f)
		{
			frequencies[i] = 0.0f;
		}
		if (frequencies[i] >= 0.0f)
		{
			if (sampleBegin == (unsigned)(-1))
				sampleBegin = (unsigned)i;
			sampleEnd = (unsigned)i+1;
		}		
		if (frequencies[i] > 0.0f)
		{
			ave_freq += frequencies[i];
			count += 1.0f;
		}
	}
	ave_freq /= count;

	for (unsigned i = sampleBegin; i < sampleEnd; i++)
		if (frequencies[i] < 0.0f) frequencies[i] = 0.0f;

}

class KeLa : public Singer
{
public:
	virtual void GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf)
	{
		if (notes.size() < 1) return; 

		char path[1024];
		sprintf(path, "KeLaSamples/%s/%s.wav", m_name.data(), lyric);

		Buffer source;
		float maxv;
		if (!ReadWavToBuffer(path, source, maxv)) return;

		unsigned freq_step = 256;
		std::vector<float> frequencies;
		float ave_freq;

		sprintf(path, "KeLaSamples/%s/%s.freq", m_name.data(), lyric);
		FILE* fp = fopen(path, "r");
		if (fp)
		{
			ave_freq = 0.0f;
			float count = 0.0f;
			while (!feof(fp))
			{
				float f;
				if (fscanf(fp, "%f", &f))
				{
					frequencies.push_back(f);
				}
				else break;

				if (f > 0.0f)
				{
					ave_freq += f;
					count += 1.0f;
				}
			}
			fclose(fp);

			ave_freq /= count;
		}
		else
		{
			DetectFreqs(source, frequencies, freq_step, ave_freq);
			fp = fopen(path, "w");
			for (size_t i = 0; i < frequencies.size(); i++)
			{
				fprintf(fp, "%f\n", frequencies[i]);
			}
			fclose(fp);
		}

		float minSampleFreq = FLT_MAX;
		float sumLen = 0.0f;
		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			if (sampleFreq < minSampleFreq) minSampleFreq = sampleFreq;
			sumLen += notes[i].fNumOfSamples;
		}

		float tempLen = 0.0f;
		std::vector<float> accLen;
		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			tempLen += notes[i].fNumOfSamples *sampleFreq / minSampleFreq;
			accLen.push_back(tempLen);
		}

		unsigned unvoicedBegin = (unsigned)(-1);
		unsigned voicedBegin = (unsigned)(-1);
		unsigned voicedEnd = (unsigned)(-1);
		unsigned unvoicedEnd = (unsigned)(-1);

		unsigned voicedBegin_id = (unsigned)(-1);
		unsigned voicedEnd_id = (unsigned)(-1);

		float firstFreq;
		float lastFreq;

		for (size_t i = 0; i < frequencies.size(); i++)
		{
			if (frequencies[i] >= 0.0f && unvoicedBegin == (unsigned)(-1))
				unvoicedBegin = i>0?(unsigned)i*freq_step - (freq_step/2):0;
			if (frequencies[i] > 0.0f && voicedBegin == (unsigned)(-1))
			{
				voicedBegin_id = (unsigned)i;
				voicedBegin = i > 0 ? (unsigned)i*freq_step - (freq_step / 2) : 0;
				firstFreq = frequencies[i];
			}

			if (frequencies[i] <= 0.0f && voicedBegin != (unsigned)(-1) && voicedEnd == (unsigned)(-1))
			{
				voicedEnd_id = (unsigned)i;
				voicedEnd = (unsigned)i*freq_step - (freq_step / 2);
				lastFreq = frequencies[i - 1];
			}

			if (frequencies[i] < 0.0f && voicedEnd != (unsigned)(-1) && unvoicedEnd == (unsigned)(-1))
			{
				unvoicedEnd = (unsigned)i*freq_step - (freq_step / 2);
				break;
			}
		}
		if (voicedEnd == (unsigned)(-1))
		{
			voicedEnd = (unsigned)source.m_data.size();
			lastFreq = frequencies[frequencies.size() - 1];
		}
		if (unvoicedEnd == (unsigned)(-1))
		{
			unvoicedEnd = (unsigned)source.m_data.size();
		}

		std::vector<SymmetricWindow> windows;
		float fPeriodCount = 0.0f;
		for (unsigned srcPos = voicedBegin; srcPos < voicedEnd; srcPos++)
		{
			float srcFreqPos = (float)srcPos / (float)freq_step;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			float srcFreq;
			if (uSrcFreqPos >= frequencies.size() - 1)
			{
				if (frequencies.size() - 1 >= voicedEnd_id) srcFreq = lastFreq;
				else srcFreq = frequencies[frequencies.size() - 1];
			}
			else
			{
				float freq1;
				if (uSrcFreqPos < voicedBegin_id) freq1 = firstFreq;
				else if (uSrcFreqPos >= voicedEnd_id) freq1 = lastFreq;
				else freq1 = frequencies[uSrcFreqPos];

				float freq2;
				if (uSrcFreqPos + 1 < voicedBegin_id) freq2 = firstFreq;
				else if (uSrcFreqPos + 1 >= voicedEnd_id) freq2 = lastFreq;
				else freq2 = frequencies[uSrcFreqPos + 1];

				srcFreq = freq1*(1.0f - fracSrcFreqPos) + freq2*fracSrcFreqPos;
			}

			unsigned winId = (unsigned)fPeriodCount;
			if (winId >= windows.size())
			{
				float srcHalfWinWidth = (float)source.m_sampleRate / srcFreq;
				Window srcWin;
				srcWin.CreateFromBuffer(source, (float)srcPos, srcHalfWinWidth);

				SymmetricWindow symWin;
				symWin.CreateFromAsymmetricWindow(srcWin);

				windows.push_back(symWin);
			}

			fPeriodCount += srcFreq / (float)source.m_sampleRate;

		}

		unsigned uTempLen = (unsigned)ceilf(tempLen);
		Buffer tempBuf;
		tempBuf.m_sampleRate = source.m_sampleRate;
		tempBuf.m_data.resize(uTempLen);
		tempBuf.SetZero();

		float rateUnvoiced0 = minSampleFreq * (float)source.m_sampleRate / firstFreq;
		float lenUnvoiced0 = (float)(voicedBegin - unvoicedBegin) / rateUnvoiced0;

		float rateUnvoiced1 = minSampleFreq * (float)source.m_sampleRate / lastFreq;
		float lenUnvoiced1 = (float)(unvoicedEnd - voicedEnd) / rateUnvoiced1;

		if (lenUnvoiced0 > tempLen)
		{
			lenUnvoiced0 = tempLen;
			lenUnvoiced1 = 0.0f;
		}

		if (lenUnvoiced0 + lenUnvoiced1 > tempLen)
			lenUnvoiced1 = tempLen - lenUnvoiced0;

		float lenVoiced = tempLen - lenUnvoiced0 - lenUnvoiced1;
		
		float tempHalfWinLen = 1.0f / minSampleFreq;

		// unvoiced 0
		float srcPos = (float)unvoicedBegin;
		bool interpolation = rateUnvoiced0 < 1.0f;
		for (unsigned uTmpPos = 0; (float)uTmpPos < lenUnvoiced0 && uTmpPos < uTempLen; uTmpPos++, srcPos += rateUnvoiced0)
		{
			float tmpValue;
			if (interpolation)
			{
				int ipos1 = (int)floorf(srcPos);
				float frac = srcPos - (float)ipos1;
				int ipos2 = ipos1 + 1;
				int ipos0 = ipos1 - 1;
				int ipos3 = ipos1 + 2;

				float p0 = source.GetSample(ipos0);
				float p1 = source.GetSample(ipos1);
				float p2 = source.GetSample(ipos2);
				float p3 = source.GetSample(ipos3);

				tmpValue = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceilf(srcPos - rateUnvoiced0*0.5f);
				int ipos2 = (int)floorf(srcPos + rateUnvoiced0*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += source.GetSample(ipos);
				}
				tmpValue = sum / (float)(ipos2 - ipos1 + 1);

			}

			float ampl = 1.0f;
			if (uTmpPos>lenUnvoiced0 - tempHalfWinLen)
			{
				float progress = (uTmpPos - (lenUnvoiced0 - tempHalfWinLen)) / tempHalfWinLen;
				ampl = (cosf(progress * (float)PI) + 1.0f)*0.5f;
			}

			tempBuf.m_data[uTmpPos] = tmpValue*ampl;
		}

		// voiced
		float fTmpWinCenter;
		unsigned noteId = 0;
		for (fTmpWinCenter = lenUnvoiced0; fTmpWinCenter < lenUnvoiced0 + lenVoiced; fTmpWinCenter += tempHalfWinLen)
		{
			float fWinId = (float)windows.size()* (fTmpWinCenter - lenUnvoiced0) / lenVoiced;
			unsigned winId0 = min((unsigned)fWinId, (unsigned)windows.size() - 1);
			unsigned winId1 = min(winId0 + 1, (unsigned)windows.size() - 1);
			float k = fWinId - (float)winId0;

			while (fTmpWinCenter > accLen[noteId]) noteId++;

			float destSampleFreq = notes[noteId].sampleFreq;
			float destHalfWinLen = 1.0f / destSampleFreq;

			SymmetricWindow& win0 = windows[winId0];
			SymmetricWindow shiftedWin0;

			SymmetricWindow& win1 = windows[winId1];
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
				l_win.m_halfWidth = destHalfWinLen;
				unsigned u_halfWidth = (unsigned)ceilf(destHalfWinLen);
				l_win.m_data.resize(u_halfWidth);

				for (unsigned i = 0; i < destHalfWinLen; i++)
					l_win.m_data[i] = (1.0f - k) *shiftedWin0.m_data[i] + k* shiftedWin1.m_data[i];
			}

			SymmetricWindow l_win2;
			SymmetricWindow *winToMerge = &l_win2;
			if (destHalfWinLen == tempHalfWinLen)
			{
				winToMerge = destWin;
			}
			else
			{
				l_win2.Scale(*destWin, tempHalfWinLen);
			}
			winToMerge->MergeToBuffer(tempBuf, fTmpWinCenter);

		}

		// unvoiced 1
		float unvoicedBegin1 = fTmpWinCenter - tempHalfWinLen;
		srcPos = (float)voicedEnd;
		for (unsigned uTmpPos = (unsigned)ceilf(unvoicedBegin1); uTmpPos < uTempLen; uTmpPos++, srcPos += rateUnvoiced1)
		{
			float tmpValue;
			if (interpolation)
			{
				int ipos1 = (int)floorf(srcPos);
				float frac = srcPos - (float)ipos1;
				int ipos2 = ipos1 + 1;
				int ipos0 = ipos1 - 1;
				int ipos3 = ipos1 + 2;

				float p0 = source.GetSample(ipos0);
				float p1 = source.GetSample(ipos1);
				float p2 = source.GetSample(ipos2);
				float p3 = source.GetSample(ipos3);

				tmpValue = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceilf(srcPos - rateUnvoiced0*0.5f);
				int ipos2 = (int)floorf(srcPos + rateUnvoiced0*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += source.GetSample(ipos);
				}
				tmpValue = sum / (float)(ipos2 - ipos1 + 1);

			}

			float ampl = 1.0f;
			if (uTmpPos < unvoicedBegin1 + tempHalfWinLen)
			{
				float progress = (uTmpPos - (unvoicedBegin1 + tempHalfWinLen)) / tempHalfWinLen;
				ampl = (cosf(progress * (float)PI) + 1.0f)*0.5f;
			}
			tempBuf.m_data[uTmpPos] += tmpValue*ampl;
		}

		// post processing
		unsigned uSumLen = (unsigned)ceilf(sumLen);
		noteBuf->m_sampleNum = uSumLen;
		noteBuf->Allocate();

		float pos_DstBuf = 0.0f;
		unsigned pos_final = 0;
		float targetPos = 0.0f;

		float multFac = m_noteVolume / maxv;

		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			targetPos += notes[i].fNumOfSamples;
			float speed = sampleFreq / minSampleFreq;

			for (; (float)pos_final < targetPos; pos_final++, pos_DstBuf += speed)
			{
				int ipos1 = (int)ceilf(pos_DstBuf - speed*0.5f);
				int ipos2 = (int)floorf(pos_DstBuf + speed*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += tempBuf.GetSample(ipos);
				}
				float value = sum / (float)(ipos2 - ipos1 + 1);

				float x2 = (float)pos_final / sumLen;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				noteBuf->m_data[pos_final] = amplitude*value*multFac;
			}

		}


	}

	void SetName(const char* name)
	{
		m_name = name;
	}

private:
	std::string m_name;
};


class KeLaFactory : public InstrumentFactory
{
public:
	KeLaFactory()
	{
#ifdef _WIN32
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		hFind = FindFirstFileA("KeLaSamples\\*", &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return;

		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY && strcmp(ffd.cFileName, ".") != 0 && strcmp(ffd.cFileName, "..") != 0)
			{
				m_SingerList.push_back(ffd.cFileName);
			}

		} while (FindNextFile(hFind, &ffd) != 0);

#else
		DIR *dir;
		struct dirent *entry;

		if (dir = opendir("KeLaSamples"))
		{
			while ((entry = readdir(dir)) != NULL)
			{
				if (entry->d_type == DT_DIR)
				{
					if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
						m_SingerList.push_back(entry->d_name);				
				}
			}
		}
#endif
	}

	virtual void GetSingerList(std::vector<std::string>& list)
	{
		list = m_SingerList;
	}

	virtual void InitiateSinger(unsigned clsInd, Singer_deferred& singer)
	{
		singer = Singer_deferred::Instance<KeLa>();
		singer.DownCast<KeLa>()->SetName(m_SingerList[clsInd].data());
	}

private:
	std::vector<std::string> m_SingerList;
};


PY_SCOREDRAFT_EXTENSION_INTERFACE GetFactory()
{
	static KeLaFactory fac;
	return &fac;
}

