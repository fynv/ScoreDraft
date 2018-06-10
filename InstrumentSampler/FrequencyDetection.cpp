#include "FrequencyDetection.h"
#include "fft.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

class Window
{
public:
	virtual ~Window(){}
	float m_halfWidth;
	std::vector<float> m_data;

	void Allocate(float halfWidth)
	{
		unsigned u_halfWidth = (unsigned)ceilf(halfWidth);
		unsigned u_width = u_halfWidth << 1;

		m_halfWidth = halfWidth;
		m_data.resize(u_width);

		SetZero();
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

		SetZero();

		int i_Center = (int)center;

		for (int i = -(int)u_halfWidth; i < (int)u_halfWidth; i++)
		{
			float window = (cosf((float)i * (float)PI / halfWidth) + 1.0f)*0.5f;

			int srcIndex = i_Center + i;
			float v_src = src.GetSample(srcIndex);

			SetSample(i, window* v_src);
		}
	}

	void MergeToBuffer(Buffer& buf, float pos)
	{
		int ipos = (int)floorf(pos);
		unsigned u_halfWidth = GetHalfWidthOfData();

		for (int i = max(-(int)u_halfWidth, -ipos); i < (int)u_halfWidth; i++)
		{
			int dstIndex = ipos + i;
			if (dstIndex >= (int)buf.m_size) break;
			buf.m_data[dstIndex] += GetSample(i);
		}
	}

	virtual unsigned GetHalfWidthOfData() const
	{
		unsigned u_width = (unsigned)m_data.size();
		unsigned u_halfWidth = u_width >> 1;

		return u_halfWidth;
	}

	virtual float GetSample(int i) const
	{
		unsigned u_width = (unsigned)m_data.size();
		unsigned u_halfWidth = u_width >> 1;

		unsigned pos;
		if (i >= 0)
		{
			if ((unsigned)i > u_halfWidth - 1) return 0.0f;
			pos = (unsigned)i;
		}
		else
		{
			if (((int)u_width + i) < (int)u_halfWidth + 1) return 0.0f;
			pos = u_width - (unsigned)(-i);
		}

		return m_data[pos];
	}

	virtual void SetSample(int i, float v)
	{
		unsigned u_width = (unsigned)m_data.size();
		unsigned u_halfWidth = u_width >> 1;

		unsigned pos;
		if (i >= 0)
		{
			if ((unsigned)i > u_halfWidth - 1) return;
			pos = (unsigned)i;
		}
		else
		{
			if (((int)u_width + i) < (int)u_halfWidth + 1) return;
			pos = u_width - (unsigned)(-i);
		}

		m_data[pos] = v;
	}

};


void fetchFrequency(unsigned length, float *samples, unsigned sampleRate, float& freq, float& dyn)
{
	unsigned len = 1;
	unsigned l = 0;
	while (len < length * 2)
	{
		l++;
		len <<= 1;
	}

	DComp* fftData = new DComp[len];
	memset(fftData, 0, sizeof(DComp)*len);

	for (unsigned i = 0; i<length; i++)
	{
		fftData[i].Re = (double)samples[i];
		fftData[i].Im = 0.0;
	}
	fft(fftData, l);

	// self-correlation
	for (unsigned i = 0; i<len; i++)
	{
		DComp c = fftData[i];
		fftData[i].Re = c.Re*c.Re + c.Im*c.Im;
		fftData[i].Im = 0.0;
	}

	ifft(fftData, l);

	dyn = (float)fftData[0].Re*700.0f;
	freq = -1.0f;

	if (fftData[0].Re > 0.01)
	{
		unsigned maxi = (unsigned)(-1);

		double lastV = fftData[0].Re;
		double maxV = 0.0f;
		bool ascending = false;

		for (unsigned i = sampleRate / 2000; i < min(sampleRate / 30, len / 2); i++)
		{
			double v = fftData[i].Re;
			if (!ascending)
			{
				if (v > lastV) ascending = true;
			}
			else
			{
				if (v < lastV)
				{
					if (fftData[i - 1].Re>maxV)
					{
						maxV = fftData[i - 1].Re;
						maxi = i - 1;
					}
					ascending = false;
				}
			}
			lastV = v;
		}

		if (maxi != (unsigned)(-1) && maxV > 0.3f* fftData[0].Re)
		{
			freq = (float)sampleRate / (float)maxi;
		}
	}

	delete[] fftData;

}

float fetchFrequency(const Buffer& buf, unsigned sampleRate)
{
	unsigned halfWinLen = 2048;
	float* temp = new float[halfWinLen * 2];

	float aveFreq = 0.0f;
	float count = 0.0f;
	for (unsigned center = 0; center < buf.m_size; center += halfWinLen)
	{
		Window win;
		win.CreateFromBuffer(buf, (float)center, (float)halfWinLen);

		for (int i = -(int)halfWinLen; i < (int)halfWinLen; i++)
			temp[i + halfWinLen] = win.GetSample(i);

		float freq;
		float dyn;
		fetchFrequency(halfWinLen * 2, temp, sampleRate, freq, dyn);

		if (freq > 0)
		{
			aveFreq += freq;
			count += 1.0f;
		}
	}

	aveFreq /= count;

	delete[] temp;

	return aveFreq;
}
