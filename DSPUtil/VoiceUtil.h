#ifndef _VoiceUtil_h
#define _VoiceUtil_h

#define Symmetric_Type_Axis 0
#define Symmetric_Type_Center 1
#define Symmetric_Type Symmetric_Type_Center

#include <vector>
#include <ReadWav.h>
#include "fft.h"

namespace VoiceUtil
{
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
				pos = (unsigned)(u_width - (unsigned)(-i));
			}

			m_data[pos] = v;
		}	

	};


	class SymmetricWindow : public Window
	{
	public:
		virtual unsigned GetHalfWidthOfData() const
		{
			unsigned u_halfWidth = (unsigned)m_data.size();
			return u_halfWidth;
		}
		virtual float GetSample(int i) const
		{
			if (i < 0)
			{
				if (-i >= m_data.size()) return 0.0f;
#if Symmetric_Type == Symmetric_Type_Axis
				return m_data[-i];
#else
				return -m_data[-i];
#endif
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
#if Symmetric_Type == Symmetric_Type_Axis
				m_data[-i] = v;
#else
				m_data[-i] = -v;
#endif

			}
		}

		void CreateFromAsymmetricWindow(const Window& src)
		{
			unsigned u_srcHalfWidth = src.GetHalfWidthOfData();
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
				fftBuf[i].Re = (double)src.GetSample((int)i);
				if (i > 0) fftBuf[fftLen - i].Re = (double)src.GetSample(-(int)(i));
			}

			fft(fftBuf, l);

			fftBuf[0].Re = 0.0f;
			fftBuf[0].Im = 0.0f;
			fftBuf[fftLen / 2].Re = 0.0f;
			fftBuf[fftLen / 2].Im = 0.0f;

			for (unsigned i = 1; i < fftLen /2; i++)
			{
				double absv = DCAbs(&fftBuf[i]);

#if Symmetric_Type == Symmetric_Type_Axis
				fftBuf[i].Re = absv;
				fftBuf[i].Im = 0.0f;
				fftBuf[fftLen-i].Re = absv;
				fftBuf[fftLen-i].Im = 0.0f;
#else
				fftBuf[i].Re = 0.0f;
				fftBuf[i].Im = absv;
				fftBuf[fftLen-i].Re = 0.0f;
				fftBuf[fftLen-i].Im = -absv;
#endif
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
#if Symmetric_Type == Symmetric_Type_Axis
					m_data[i] += src.m_data[uSrcPos];	
#else
					m_data[i] -= src.m_data[uSrcPos];
#endif
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

	};




}

#endif 
