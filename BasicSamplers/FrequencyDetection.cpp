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

float fetchFrequency(const Buffer& buf, unsigned sampleRate)
{
	unsigned l = 12;
	unsigned halfWinLen = 1 << (l - 1);

	float* fft_acc = new float[halfWinLen];
	memset(fft_acc, 0, sizeof(float)*halfWinLen);

	DComp* fftData = new DComp[halfWinLen * 2];
	fftData[0].Re = 0.0;
	fftData[0].Im = 0.0;

	for (unsigned center = 0; center < buf.m_size; center += halfWinLen)
	{
		Window win;
		win.CreateFromBuffer(buf, (float)center, (float)halfWinLen);

		for (unsigned i = 1; i<halfWinLen * 2; i++)
		{
			fftData[i].Re = (double)win.GetSample(i - halfWinLen);
			fftData[i].Im = 0.0;
		}
		fft(fftData, l);

		for (unsigned i = 0; i < halfWinLen; i++)
		{
			DComp c = fftData[i];
			fft_acc[i] += c.Re*c.Re + c.Im*c.Im;
		}
	}

	for (unsigned i = 0; i < halfWinLen; i++)
	{
		fftData[i].Re = (double)fft_acc[i];
		fftData[i].Im = 0.0;
	}
	ifft(fftData, l);

	unsigned maxi = (unsigned)(-1);

	double lastV = fftData[0].Re;
	double maxV = 0.0f;
	bool ascending = false;

	for (unsigned i = sampleRate / 2000; i < min(sampleRate / 30, halfWinLen); i++)
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

	float freq = (float)sampleRate / (float)maxi;

	delete[] fft_acc;

	return freq;
}
