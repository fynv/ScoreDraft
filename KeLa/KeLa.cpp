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
#include "fft.h"
#include "FrequencyDetection.h"
#include <float.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

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

struct Window
{
	float m_halfWidth;
	std::vector<float> m_data;

	float GetSample(int i) const
	{
		size_t u_width = m_data.size();
		if (i < -(int)u_width || i >= (int)u_width) return 0.0f;
		unsigned pos = (unsigned)(i + (int)u_width) % u_width;
		return m_data[pos];
	}

	void SetSample(int i, float v)
	{
		size_t u_width = m_data.size();
		if (i < -(int)u_width || i >= (int)u_width) return;
		unsigned pos = (unsigned)(i + (int)u_width) % u_width;
		m_data[pos] = v;
	}

	void SetZero()
	{
		memset(m_data.data(), 0, sizeof(float)*m_data.size());
	}
};

void HanningWindow(const Buffer& src, float center, float halfWidth, Window& dst)
{
	unsigned u_halfWidth = (unsigned)ceilf(halfWidth);
	unsigned u_width = u_halfWidth << 1;

	dst.m_halfWidth = halfWidth;
	dst.m_data.resize(u_width);

	unsigned u_Center = (unsigned)center;

	for (int i = -(int)u_halfWidth; i < (int)u_halfWidth; i++)
	{
		float window = (cosf((float)i * (float)PI / halfWidth) + 1.0f)*0.5f;

		int srcIndex = (int)u_Center + i;
		float v_src = src.GetSample(srcIndex);

		dst.SetSample(i, window* v_src);
	}
}

void MergeWindowToBuffer(const Window& win, Buffer& buf, float pos)
{
	int ipos = (int)pos;
	unsigned u_halfWidth = (unsigned)ceilf(win.m_halfWidth);
	unsigned u_width = u_halfWidth << 1;

	for (int i = max(-(int)u_halfWidth, -ipos); i < (int)u_halfWidth; i++)
	{
		int dstIndex = ipos + i;
		if (dstIndex >= (int)buf.m_data.size()) break;
		buf.m_data[dstIndex] += win.GetSample(i);
	}
}

void DetectFreqs(const Buffer& buf, std::vector<float>& frequencies, unsigned step, float& ave_freq)
{
	unsigned halfWinLen = 512;
	float* temp = new float[halfWinLen * 2];

	for (unsigned center = 0; center < buf.m_data.size(); center += step)
	{
		Window win;
		HanningWindow(buf, (float)center, (float)halfWinLen, win);

		for (int i = -(int)halfWinLen; i < (int)halfWinLen; i++)
			temp[i + halfWinLen] = win.GetSample(i);

		bool suc;
		float freq = fetchFrequency(halfWinLen * 2, temp, buf.m_sampleRate, suc);

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
		if (frequencies[i] != 0)
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
	for (size_t i = 0; i < frequencies.size(); i++)
	{
		if (i<BestRange.begin || i>BestRange.end)
			frequencies[i] = 0.0f;
		if (frequencies[i] > 0.0f)
		{
			ave_freq += frequencies[i];
			count += 1.0f;
		}
	}
	ave_freq /= count;
}

class KeLa : public Singer
{
public:
	virtual void GenerateWave(const char* lyric, std::vector<SingerNoteParams> notes, VoiceBuffer* noteBuf)
	{
		char path[1024];
		sprintf(path, "KeLaSamples/%s.wav", lyric);

		Buffer source;
		float maxv;
		if (!ReadWavToBuffer(path, source, maxv)) return;

		unsigned freq_step = 256;
		std::vector<float> frequencies;
		float ave_freq;

		sprintf(path, "KeLaSamples/%s.freq", lyric);
		FILE* fp = fopen(path, "r");
		if (fp)
		{
			fscanf(fp, "%f", &ave_freq);
			while (!feof(fp))
			{
				float f;
				if (fscanf(fp, "%f", &f))
				{
					frequencies.push_back(f);
				}
				else break;
			}
			fclose(fp);
		}
		else
		{
			DetectFreqs(source, frequencies, freq_step, ave_freq);
			fp = fopen(path, "w");
			fprintf(fp, "%f\n", ave_freq);

			for (size_t i = 0; i < frequencies.size(); i++)
			{
				fprintf(fp, "%f\n", frequencies[i]);
			}

			fclose(fp);
		}

		float minSampleFreq = FLT_MAX;
		float trueSumLen = 0.0f;
		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			if (sampleFreq < minSampleFreq) minSampleFreq = sampleFreq;
			trueSumLen += notes[i].fNumOfSamples;
		}

		float sumLen = 0.0f;
		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			sumLen += notes[i].fNumOfSamples *sampleFreq / minSampleFreq;
		}
		
		Buffer bufResampled;
		bufResampled.m_sampleRate = source.m_sampleRate;

		unsigned voiceBegin = (unsigned)(-1);
		unsigned voiceEnd = (unsigned)(-1);

		float srcPos = 0.0f;
		while ((unsigned)srcPos < source.m_data.size())
		{
			float srcFreqPos = srcPos / (float)freq_step;
			unsigned uSrcFreqPos = (unsigned)srcFreqPos;
			float fracSrcFreqPos = srcFreqPos - (float)uSrcFreqPos;

			bool voicing;
			{
				unsigned nearestSrcFreqPos = min((unsigned)(srcFreqPos + 0.5f), frequencies.size() - 1);
				voicing = frequencies[nearestSrcFreqPos] > 0.0f;
			}
			if (voicing)
			{
				if (voiceBegin == (unsigned)(-1)) voiceBegin = (unsigned)bufResampled.m_data.size();
			}
			else if (voiceBegin != (unsigned)(-1))
			{
				if (voiceEnd == (unsigned)(-1)) voiceEnd = (unsigned)bufResampled.m_data.size();
			}

			float srcFreq;
			if (uSrcFreqPos >= frequencies.size() - 1)
			{
				srcFreq = frequencies[frequencies.size() - 1];
				if (srcFreq == 0.0f) srcFreq = ave_freq;
			}
			else
			{
				float freq1 = frequencies[uSrcFreqPos];
				if (freq1 == 0.0f) freq1 = ave_freq;
				float freq2 = frequencies[uSrcFreqPos + 1];
				if (freq2 == 0.0f) freq2 = ave_freq;
				srcFreq = freq1*(1.0f - fracSrcFreqPos) + freq2*fracSrcFreqPos;
			}

			float speed = minSampleFreq* (float)source.m_sampleRate / srcFreq;
			bool interpolation = speed < 1.0f;

			float dstValue;

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

				dstValue = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceilf(srcPos - speed*0.5f);
				int ipos2 = (int)floorf(srcPos + speed*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += source.GetSample(ipos);
				}
				dstValue = sum / (float)(ipos2 - ipos1 + 1);

			}
			bufResampled.m_data.push_back(dstValue);

			srcPos += speed;
		}
		if (voiceEnd == (unsigned)(-1)) voiceEnd = (unsigned)bufResampled.m_data.size();

		// CreateWindows

		float halfWinWidth = 1.0f / minSampleFreq;

		std::vector<Window> windows;
		float center;
		for (center = (float)voiceBegin; center < (float)voiceEnd; center += halfWinWidth)
		{
			Window win;
			HanningWindow(bufResampled, center, halfWinWidth, win);
			windows.push_back(win);
		}

		voiceEnd = (unsigned)(center - halfWinWidth);

		unsigned VoicedLen = voiceEnd - voiceBegin + 1;
		unsigned UnvoidedLen = (unsigned)bufResampled.m_data.size() - VoicedLen;

		float stretch_rate = (sumLen - (float)UnvoidedLen) / (float)VoicedLen;

		unsigned destLen = (unsigned)ceilf(sumLen);

		Buffer DestBuf;
		DestBuf.m_sampleRate = bufResampled.m_sampleRate;
		DestBuf.m_data.resize(destLen);
		DestBuf.SetZero();

		unsigned destPos = 0;
		for (; destPos < destLen && destPos < voiceBegin; destPos++)
		{
			float srcV = bufResampled.m_data[destPos];

			if ((float)(voiceBegin - destPos) < halfWinWidth)
			{
				float window = (cosf(((float)destPos + halfWinWidth - (float)voiceBegin) * (float)PI / halfWinWidth) + 1.0f)*0.5f;
				DestBuf.m_data[destPos] = srcV*window;
			}
			else
			{
				DestBuf.m_data[destPos] = srcV;
			}
		}

		if (stretch_rate > 0.0f)
		{
			unsigned numWin = (unsigned)((float)windows.size()*stretch_rate);
			float winCenter = (float)destPos;
			for (unsigned w = 0; w < numWin; w++, winCenter += halfWinWidth)
			{
				float fsrcw = w / stretch_rate;
				unsigned usrcw = (unsigned)fsrcw;
				float frac = fsrcw - (float)usrcw;

				Window l_win;
				Window* winToMerge = &l_win;

				if (usrcw >= windows.size() - 1)
					winToMerge = &windows[windows.size() - 1];
				else
				{
					Window& win1 = windows[usrcw];
					Window& win2 = windows[usrcw + 1];

					l_win.m_halfWidth = halfWinWidth;
					unsigned u_HalfWidth = (unsigned)ceilf(halfWinWidth);
					unsigned u_Width = u_HalfWidth << 1;

					l_win.m_data.resize(u_Width);

					for (int i = -(int)u_HalfWidth; i < (int)u_HalfWidth; i++)
					{
						float v1 = win1.GetSample(i);
						float v2 = win2.GetSample(i);
						l_win.SetSample(i, v1*(1.0f - frac) + v2*frac);
					}
				}
				MergeWindowToBuffer(*winToMerge, DestBuf, winCenter);
			}

			destPos = (unsigned)(winCenter - halfWinWidth);
		}

		for (unsigned srcPos = voiceEnd; destPos < destLen && srcPos < (unsigned)bufResampled.m_data.size(); srcPos++, destPos++)
		{
			float srcV = bufResampled.m_data[srcPos];
			if ((float)(srcPos - voiceEnd)< halfWinWidth)
			{
				float window = (cosf(((float)(srcPos - voiceEnd) - halfWinWidth) * (float)PI / halfWinWidth) + 1.0f)*0.5f;
				DestBuf.m_data[destPos] = srcV*window;
			}
			else
			{
				DestBuf.m_data[destPos] = srcV;
			}
		}

		// post processing
		unsigned finalSize = (unsigned)ceilf(trueSumLen);
		noteBuf->m_sampleNum = finalSize;
		noteBuf->Allocate();

		float pos_DstBuf = 0.0f;
		unsigned pos_final = 0;
		float targetPos = 0.0f;

		for (size_t i = 0; i < notes.size(); i++)
		{
			float sampleFreq = notes[i].sampleFreq;
			targetPos += notes[i].fNumOfSamples;
			float speed = sampleFreq / minSampleFreq;

			for (; (float)pos_final < targetPos; pos_final++, pos_DstBuf+=speed)
			{
				int ipos1 = (int)ceilf(pos_DstBuf - speed*0.5f);
				int ipos2 = (int)floorf(pos_DstBuf + speed*0.5f);

				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += DestBuf.GetSample(ipos);
				}
				float value = sum / (float)(ipos2 - ipos1 + 1);

				float x2 = (float)pos_final / trueSumLen;
				float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

				noteBuf->m_data[pos_final] = amplitude*value*m_noteVolume;
			}


		}
	}
};


PY_SCOREDRAFT_EXTENSION_INTERFACE GetFactory()
{
	static TypicalInstrumentFactory fac;
	fac.AddSinger<KeLa>("KeLa");
	return &fac;
}

