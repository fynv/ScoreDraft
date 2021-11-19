#ifndef _VoiceUtil_h
#define _VoiceUtil_h

#define Symmetric_Type_Axis 0
#define Symmetric_Type_Center 1
#define Symmetric_Type Symmetric_Type_Center


#include <vector>
#include "fft.h"
#include <stdlib.h>
#include <memory.h>
#include <cmath>
#include <float.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


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



namespace VoiceUtil
{
	inline void calcPOT(unsigned lenIn, unsigned& lenOut, unsigned& l)
	{
		lenOut = 1;
		l = 0;
		while (lenOut < lenIn)
		{
			l++;
			lenOut <<= 1;
		}
	}

	struct Buffer
	{
		unsigned m_sampleRate;
		std::vector<float> m_data;

		void Allocate(unsigned size)
		{
			m_data.resize(size);
			SetZero();
		}

		float GetSample(int i) const
		{
			size_t usize = m_data.size();
			if (i<0 || i >= (int)usize) return 0.0f;
			return m_data[i];
		}

		void SetZero()
		{
			memset(m_data.data(), 0, sizeof(float)*m_data.size());
		}

		void SetSample(int i, float v)
		{
			size_t usize = m_data.size();
			if (i < 0 || i >= (int)usize) return;
			m_data[i] = v;
		}
		
		void AddToSample(int i, float v)
		{
			size_t usize = m_data.size();
			if (i < 0 || i >= (int)usize) return;
			m_data[i] += v;
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

	};

	class AmpSpectrum;
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
				if (dstIndex >= (int)buf.m_data.size()) break;
				buf.m_data[dstIndex] += GetSample(i);
			}
		}

		virtual void Interpolate(const Window& win0, const Window& win1, float k, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned u_halfWidth = (unsigned)ceilf(targetHalfWidth);
			unsigned u_Width = u_halfWidth << 1;
			m_data.resize(u_Width);

			for (int i = -((int)u_halfWidth - 1); i <= (int)u_halfWidth - 1; i++)
			{
				float v0 = win0.GetSample(i);
				float v1 = win1.GetSample(i);
				this->SetSample(i, (1.0f - k) *v0 + k* v1);
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

		virtual void Scale(const Window& src, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
			unsigned u_TargetWidth = u_TargetHalfWidth << 1;
			m_data.resize(u_TargetWidth);

			float rate = src.m_halfWidth / targetHalfWidth;
			bool interpolation = rate < 1.0f;
			for (int i = -(int)(u_TargetHalfWidth - 1); i <= (int)(u_TargetHalfWidth - 1); i++)
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

				this->SetSample(i, destValue);

			}
		}

		inline void CreateFromAmpSpec_noise(const AmpSpectrum& src, float targetHalfWidth=-1.0f);

	};

	class AmpSpectrum
	{
	public:
		float m_halfWidth;
		std::vector<float> m_data;

		void Allocate(float halfWidth)
		{
			m_halfWidth = halfWidth;
			unsigned specLen = (unsigned)ceilf(m_halfWidth*0.5f);
			m_data.resize(specLen);

			SetZero();
		}

		void SetZero()
		{
			memset(m_data.data(), 0, sizeof(float)*m_data.size());
		}

		bool NonZero()
		{
			for (unsigned i = 1; i < (unsigned)m_data.size(); i++)
				if (m_data[i] != 0.0f) return true;
			return false;
		}

		void CreateFromWindow(const Window& src)
		{
			m_halfWidth = src.m_halfWidth;
			unsigned u_srcHalfWidth = src.GetHalfWidthOfData();
			unsigned l = 0;
			unsigned fftLen = 1;
			while (fftLen < u_srcHalfWidth)
			{
				l++;
				fftLen <<= 1;
			}
			float fLen = (float)fftLen;

			Window l_scaled;
			const Window* scaled = &l_scaled;
			if (src.m_halfWidth == fLen)
			{
				scaled = &src;
			}
			else
			{
				l_scaled.Scale(src, fLen);
			}

			DComp* fftBuf = new DComp[fftLen];
			memset(fftBuf, 0, sizeof(DComp)*fftLen);

			for (unsigned i = 0; i < fftLen; i++)
			{
				fftBuf[i].Re = (double)scaled->GetSample((int)i) + (double)scaled->GetSample((int)i - (int)fftLen);
			}

			fft(fftBuf, l);

			float rate = m_halfWidth / fLen;
			m_data.resize((unsigned)ceilf(m_halfWidth*0.5f));
			for (unsigned i = 0; i < (unsigned)m_data.size(); i++)
			{
				if (i >= fftLen)
					m_data[i] = 0.0f;
				else
					m_data[i] = (float)DCAbs(&fftBuf[i])*rate;
			}
			delete[] fftBuf;
		}

		void Interpolate(const AmpSpectrum& spec0, const AmpSpectrum& spec1, float k, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned specLen = (unsigned)ceilf(m_halfWidth*0.5f);
			m_data.resize(specLen);

			for (unsigned i = 0; i <specLen; i++)
			{
				float v0 = spec0.GetSample(i);
				float v1 = spec1.GetSample(i);
				m_data[i]= (1.0f - k) *v0 + k* v1;
			}
		}


		float GetSample(int i) const
		{
			if (i < 0) i = 0;
			if (i >= (int)m_data.size()) i = (int)m_data.size() - 1;
			return m_data[i];
		}

		void Scale(const AmpSpectrum& src, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned specLen = (unsigned)ceilf(m_halfWidth*0.5f);
			m_data.resize(specLen);

			float rate = src.m_halfWidth / targetHalfWidth;
			float mulRate = sqrtf(targetHalfWidth / src.m_halfWidth);
			bool interpolation = rate < 1.0f;
			for (unsigned i = 0; i < specLen; i++)
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

				m_data[i] = destValue*mulRate;

			}
		}

	};

	void Window::CreateFromAmpSpec_noise(const AmpSpectrum& src, float targetHalfWidth)
	{
		unsigned l = 0;
		unsigned fftLen = 1;
		while ((float)fftLen < src.m_halfWidth)
		{
			l++;
			fftLen <<= 1;
		}

		DComp* fftBuf = new DComp[fftLen];
		memset(fftBuf, 0, sizeof(DComp)*fftLen);

		float rate = (float)fftLen / src.m_halfWidth;

		for (unsigned i = 1; i < (unsigned)src.m_data.size(); i++)
		{
			if (i < fftLen / 2)
			{
				float angle = (float)rand01()*(float)PI*2.0f;
				float re = src.m_data[i] * cosf(angle) * rate;
				float im = src.m_data[i] * sinf(angle) * rate;

				fftBuf[i].Re = (double)re;
				fftBuf[i].Im = (double)im;

				fftBuf[fftLen - i].Re = (double)re;
				fftBuf[fftLen - i].Im = -(double)im;
				
			}
		}

		ifft(fftBuf, l);

		Window tempWin;
		tempWin.m_halfWidth = (float)fftLen;
		tempWin.m_data.resize(fftLen * 2);

		for (unsigned i = 0; i < fftLen; i++)
		{
			float window = (cosf((float)i * (float)PI / tempWin.m_halfWidth) + 1.0f)*0.5f;
			tempWin.m_data[i] = window*(float)fftBuf[i].Re;
			if (i>0)
				tempWin.m_data[fftLen * 2 - i] = window*(float)fftBuf[fftLen - i].Re;
		}
		delete[] fftBuf;

		this->Scale(tempWin, targetHalfWidth>0.0f ? targetHalfWidth: src.m_halfWidth);

	}

	class SymmetricWindow_Base : public Window
	{
	public:
		virtual ~SymmetricWindow_Base(){}
		bool NonZero()
		{
			for (unsigned i = 0; i < (unsigned)m_data.size(); i++)
				if (m_data[i] != 0.0f) return true;
			return false;
		}

		virtual unsigned GetHalfWidthOfData() const
		{
			unsigned u_halfWidth = (unsigned)m_data.size();
			return u_halfWidth;
		}

		virtual void Scale(const SymmetricWindow_Base& src, float targetHalfWidth)
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

		virtual void Interpolate(const SymmetricWindow_Base& win0, const SymmetricWindow_Base& win1, float k, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned u_halfWidth = (unsigned)ceilf(targetHalfWidth);
			m_data.resize(u_halfWidth);

			for (unsigned i = 0; i <= u_halfWidth - 1; i++)
			{
				float v0 = win0.GetSample(i);
				float v1 = win1.GetSample(i);
				this->SetSample(i, (1.0f - k) *v0 + k* v1);
			}
		}

	};


	class SymmetricWindow_Axis : public SymmetricWindow_Base
	{
	public:
		virtual ~SymmetricWindow_Axis(){}
		virtual float GetSample(int i) const
		{
			if (i < 0)
			{
				if (-i >= m_data.size()) return 0.0f;
				return m_data[-i];
			}
			else
			{
				if (i >= m_data.size()) return 0.0f;
				return m_data[i];
			}			
		}
		virtual void SetSample(int i, float v)
		{
			if (i < 0)
			{
				if (-i >= m_data.size()) return;
				m_data[-i] = v;
			}
			else
			{
				if (i >= m_data.size()) return;
				m_data[i] = v;
			}
		}

		void CreateFromAsymmetricWindow(const Window& src)
		{
			unsigned u_srcHalfWidth = src.GetHalfWidthOfData();
	
			unsigned l = 0;
			unsigned fftLen = 1;
			while (fftLen < u_srcHalfWidth)
			{
				l++;
				fftLen <<= 1;
			}

			DComp* fftBuf = new DComp[fftLen];
			memset(fftBuf, 0, sizeof(DComp)*fftLen);

			for (unsigned i = 0; i < fftLen; i++)
			{
				fftBuf[i].Re = (double)src.GetSample((int)i) + (double)src.GetSample((int)i - (int)fftLen);
			}

			fft(fftBuf, l);

			fftBuf[0].Re = 0.0f;
			fftBuf[0].Im = 0.0f;
			fftBuf[fftLen / 2].Re = 0.0f;
			fftBuf[fftLen / 2].Im = 0.0f;

			for (unsigned i = 1; i < fftLen /2; i++)
			{
				double absv = DCAbs(&fftBuf[i]);

				fftBuf[i].Re = absv;
				fftBuf[i].Im = 0.0f;
				fftBuf[fftLen-i].Re = absv;
				fftBuf[fftLen-i].Im = 0.0f;
			}

			ifft(fftBuf, l);

			m_data.resize(fftLen);
			m_halfWidth = (float)(fftLen);
			float rate = m_halfWidth /  src.m_halfWidth;

			for (unsigned i = 0; i < fftLen; i++)
				m_data[i] = (float)fftBuf[i].Re;

		
			// rewindow
			float amplitude = sqrtf(rate);
			for (unsigned i = 0; i < fftLen; i++)
			{
				float window = (cosf((float)i * (float)PI / m_halfWidth) + 1.0f)*0.5f;
				m_data[i] *= window*amplitude;
			}

			delete[] fftBuf;

		}

		void Repitch_FormantPreserved(const SymmetricWindow_Axis& src, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
			m_data.resize(u_TargetHalfWidth);

			float srcHalfWidth = src.m_halfWidth;
			unsigned uSrcHalfWidth = src.GetHalfWidthOfData();
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
					srcPos += targetHalfWidth;
					uSrcPos = (unsigned)(srcPos + 0.5f);
				}

				srcPos = targetHalfWidth - (float)i;
				uSrcPos = (unsigned)(srcPos + 0.5f);

				while (uSrcPos < uSrcHalfWidth)
				{
					m_data[i] += src.m_data[uSrcPos];	
					srcPos += targetHalfWidth;
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

		void CreateFromAmpSpec(const AmpSpectrum& src, float targetHalfWidth=-1.0f)
		{
			unsigned l = 0;
			unsigned fftLen = 1;
			while ((float)fftLen < src.m_halfWidth)
			{
				l++;
				fftLen <<= 1;
			}

			DComp* fftBuf = new DComp[fftLen];
			memset(fftBuf, 0, sizeof(DComp)*fftLen);

			float rate = (float)fftLen / src.m_halfWidth;

			for (unsigned i = 0; i < (unsigned)src.m_data.size(); i++)
			{
				if (i < fftLen / 2)
				{
					fftBuf[i].Re = src.m_data[i] * rate;
					if (i>0)
						fftBuf[fftLen - i].Re = src.m_data[i] * rate;
				}
			}

			ifft(fftBuf, l);

			SymmetricWindow_Axis tempWin;
			tempWin.m_halfWidth = (float)fftLen;
			tempWin.m_data.resize(fftLen);

			for (unsigned i = 0; i < fftLen; i++)
			{
				float window = (cosf((float)i * (float)PI / tempWin.m_halfWidth) + 1.0f)*0.5f;
				tempWin.m_data[i] = window*(float)fftBuf[i].Re;
			}
			delete[] fftBuf;

			this->Scale(tempWin, targetHalfWidth>0.0f ? targetHalfWidth: src.m_halfWidth);

		}

	};


	class SymmetricWindow_Center : public SymmetricWindow_Base
	{
	public:
		virtual ~SymmetricWindow_Center(){}
		virtual float GetSample(int i) const
		{
			if (i < 0)
			{
				if (-i >= m_data.size()) return 0.0f;
				return -m_data[-i];
			}
			else
			{
				if (i >= m_data.size()) return 0.0f;
				return m_data[i];
			}
		}
		virtual void SetSample(int i, float v)
		{
			if (i < 0)
			{
				if (-i >= m_data.size()) return;
				m_data[-i] = -v;
			}
			else
			{
				if (i >= m_data.size()) return;
				m_data[i] = v;
			}
		}

		void CreateFromAsymmetricWindow(const Window& src)
		{
			unsigned u_srcHalfWidth = src.GetHalfWidthOfData();

			unsigned l = 0;
			unsigned fftLen = 1;
			while (fftLen < u_srcHalfWidth)
			{
				l++;
				fftLen <<= 1;
			}

			DComp* fftBuf = new DComp[fftLen];
			memset(fftBuf, 0, sizeof(DComp)*fftLen);

			for (unsigned i = 0; i < fftLen; i++)
			{
				fftBuf[i].Re = (double)src.GetSample((int)i) + (double)src.GetSample((int)i - (int)fftLen);
			}

			fft(fftBuf, l);

			fftBuf[0].Re = 0.0f;
			fftBuf[0].Im = 0.0f;
			fftBuf[fftLen / 2].Re = 0.0f;
			fftBuf[fftLen / 2].Im = 0.0f;

			for (unsigned i = 1; i < fftLen / 2; i++)
			{
				double absv = DCAbs(&fftBuf[i]);
				fftBuf[i].Re = 0.0f;
				fftBuf[i].Im = absv;
				fftBuf[fftLen - i].Re = 0.0f;
				fftBuf[fftLen - i].Im = -absv;
			}

			ifft(fftBuf, l);

			m_data.resize(fftLen);
			m_halfWidth = (float)(fftLen);
			float rate = m_halfWidth / src.m_halfWidth;

			for (unsigned i = 0; i < fftLen; i++)
				m_data[i] = (float)fftBuf[i].Re;


			// rewindow
			float amplitude = sqrtf(rate);
			for (unsigned i = 0; i < fftLen; i++)
			{
				float window = (cosf((float)i * (float)PI / m_halfWidth) + 1.0f)*0.5f;
				m_data[i] *= window*amplitude;
			}

			delete[] fftBuf;

		}

		void Repitch_FormantPreserved(const SymmetricWindow_Center& src, float targetHalfWidth)
		{
			m_halfWidth = targetHalfWidth;
			unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
			m_data.resize(u_TargetHalfWidth);

			float srcHalfWidth = src.m_halfWidth;
			unsigned uSrcHalfWidth = src.GetHalfWidthOfData();
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
					srcPos += targetHalfWidth;
					uSrcPos = (unsigned)(srcPos + 0.5f);
				}

				srcPos = targetHalfWidth - (float)i;
				uSrcPos = (unsigned)(srcPos + 0.5f);

				while (uSrcPos < uSrcHalfWidth)
				{
					m_data[i] -= src.m_data[uSrcPos];
					srcPos += targetHalfWidth;
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

		void CreateFromAmpSpec(const AmpSpectrum& src, float targetHalfWidth = -1.0f)
		{
			unsigned l = 0;
			unsigned fftLen = 1;
			while ((float)fftLen < src.m_halfWidth)
			{
				l++;
				fftLen <<= 1;
			}

			DComp* fftBuf = new DComp[fftLen];
			memset(fftBuf, 0, sizeof(DComp)*fftLen);

			float rate = (float)fftLen / src.m_halfWidth;

			for (unsigned i = 1; i < (unsigned)src.m_data.size(); i++)
			{
				if (i < fftLen / 2)
				{
					fftBuf[i].Im = src.m_data[i] * rate;
					fftBuf[fftLen - i].Im = -src.m_data[i] * rate;
				}
			}

			ifft(fftBuf, l);

			SymmetricWindow_Center tempWin;
			tempWin.m_halfWidth = (float)fftLen;
			tempWin.m_data.resize(fftLen);

			for (unsigned i = 0; i < fftLen; i++)
			{
				float window = (cosf((float)i * (float)PI / tempWin.m_halfWidth) + 1.0f)*0.5f;
				tempWin.m_data[i] = window*(float)fftBuf[i].Re;
			}
			delete[] fftBuf;

			this->Scale(tempWin, targetHalfWidth>0.0f ? targetHalfWidth:src.m_halfWidth);

		}

	};

#if Symmetric_Type==Symmetric_Type_Axis
	typedef SymmetricWindow_Axis SymmetricWindow;
#else
	typedef SymmetricWindow_Center SymmetricWindow;
#endif


#define AutoRegression_numCoefs 10
#define AutoRegression_numSD 10

	class AutoRegression
	{
		static void LevinsonSolve(const float* samples, float* solution, unsigned n)
		{
			float *solution_last = new float[n];

			for (unsigned pass = 0; pass < n; pass++)
			{
				float numerator = samples[pass + 1];
				float denominator = samples[0];

				for (unsigned i = 0; i < pass; i++)
				{
					numerator -= solution_last[i] * samples[pass - i];
					denominator -= solution_last[i] * samples[i + 1];
				}

				float solution_pass = numerator / denominator;
				solution[pass] = solution_pass;
				for (unsigned i = 0; i < pass; i++)
					solution[i] = solution_last[i]-solution_pass*solution_last[pass - 1 - i];

				memcpy(solution_last, solution, sizeof(float)*n);
			}

			delete[] solution_last;

			/*for (unsigned i = 0; i < n; i++)
			{
				float y = samples[i + 1];
				float y1 = 0.0f;
				for (unsigned j = 0; j < n; j++)
				{
					int delta = (int)i - (int)j;
					if (delta < 0) delta = -delta;
					y1 += samples[delta] * solution[j];
				}
				printf("%f %f\n", y, y1);
			}*/
		}

	public:
		float m_coefs[AutoRegression_numCoefs];
		float m_sd[AutoRegression_numSD];

		void Estimate(const float* samples, unsigned count)
		{
			if (count <= AutoRegression_numCoefs)
				return;

			float ave = 0.0f;
			for (unsigned i = 0; i < count; i++)
			{
				ave += samples[i];
			}
			ave /= (float)(count);

			float* _samples = new float[count];
			for (unsigned i = 0; i < count; i++)
			{
				_samples[i] = samples[i] - ave;
			}

			float ac[AutoRegression_numCoefs + 1];
			for (unsigned i = 0; i <= AutoRegression_numCoefs; i++)
			{
				float sum = 0.0f;
				for (unsigned j = 0; j < count - i; j++)
				{
					sum += _samples[j] * _samples[j + i];
				}
				ac[i] = sum;
			}

			if (ac[0] == 0.0f)
			{
				memset(m_coefs, 0, sizeof(float)* AutoRegression_numCoefs);
				memset(m_sd, 0, sizeof(float)* AutoRegression_numSD);
				return;
			}

			LevinsonSolve(ac, m_coefs, AutoRegression_numCoefs);

			unsigned num_samples = count - AutoRegression_numCoefs;

			float pos = 0.0f;
			float step = (float)num_samples / AutoRegression_numSD;
			for (unsigned j = 0; j < AutoRegression_numSD; j++, pos += step)
			{
				float dev = 0.0f;
				float sqrErr = 0.0f;
				for (unsigned i = (unsigned)ceilf(pos); i <= (unsigned)floorf(pos + step); i++)
				{
					if (i + AutoRegression_numCoefs < count)
					{
						float est = 0.0f;
						for (unsigned j = 0; j < AutoRegression_numCoefs; j++)
						{
							est += m_coefs[AutoRegression_numCoefs-1-j] * _samples[i + j];
						}
						float err_i = _samples[i + AutoRegression_numCoefs] - est;
						sqrErr += err_i*err_i;
						dev += 1.0f;
					}
				}

				m_sd[j] = sqrtf(sqrErr / dev);

				//printf("%f ", m_sd[j]);
			}
			//printf("\n");

			delete[] _samples;
		}

	};


}

#endif 
