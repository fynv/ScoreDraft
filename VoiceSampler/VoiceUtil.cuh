#include "fft.cuh"
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


inline __device__ float d_WndGetSample(const float* s_Wnd, unsigned u_halfWidth, int i)
{
	if (i< -(int)(u_halfWidth - 1) || i>(int)(u_halfWidth - 1)) return 0.0f;
	int idst = i >= 0 ? i : ((int)u_halfWidth * 2 + i);
	return s_Wnd[idst];
}

inline __device__ void d_WndSetSample(float* s_Wnd, unsigned u_halfWidth, int i, float v)
{
	if (i< -(int)(u_halfWidth - 1) || i>(int)(u_halfWidth - 1)) return;
	int idst = i >= 0 ? i : ((int)u_halfWidth * 2 + i);
	s_Wnd[idst] = v;
}

__device__ void d_captureFromBuf(unsigned srcLen, float* d_srcBuf, unsigned srcPos, float halfWinlen, unsigned u_halfWidth, float* s_captureWnd)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;
	for (int i = (int)workerId - (int)(u_halfWidth - 1); i <= (int)(u_halfWidth -1); i += (int)numWorker)
	{
		int isrc = (int)srcPos + i;
		float v = 0.0f;
		if (isrc >= 0 && isrc < srcLen)
			v = d_srcBuf[isrc];

		v *= (cosf((float)i * PI / halfWinlen) + 1.0f)*0.5f;

		d_WndSetSample(s_captureWnd, u_halfWidth, i, v);
	}
	if (workerId==0)
		s_captureWnd[u_halfWidth] = 0.0f;

	__syncthreads();

}

__device__ void d_ScaleWindow(float srcHalfWinlen, unsigned u_SrcHalfWidth, const float* s_Wnd, float* s_dstWnd, float targetHalfWidth)
{
	unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
	float rate = srcHalfWinlen / targetHalfWidth;
	bool interpolation = rate < 1.0f;

	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	for (int i = (int)workerId - (int)(u_TargetHalfWidth - 1); i <= (int)(u_TargetHalfWidth - 1); i += (int)numWorker)
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

			float p0 = d_WndGetSample(s_Wnd, u_SrcHalfWidth, ipos0);
			float p1 = d_WndGetSample(s_Wnd, u_SrcHalfWidth, ipos1);
			float p2 = d_WndGetSample(s_Wnd, u_SrcHalfWidth, ipos2);
			float p3 = d_WndGetSample(s_Wnd, u_SrcHalfWidth, ipos3);

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
				sum += d_WndGetSample(s_Wnd, u_SrcHalfWidth, ipos);
			}
			destValue = sum / (float)(ipos2 - ipos1 + 1);
		}
		d_WndSetSample(s_dstWnd, u_TargetHalfWidth, i, destValue);
	}
	if (workerId == 0)
		s_dstWnd[u_TargetHalfWidth] = 0.0f;
	__syncthreads();
}

__device__ void d_CreateAmpSpectrumFromWindow(float halfWinlen, unsigned u_halfWidth, float* s_Wnd, float* s_res_wnd, unsigned uSpecLen)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	unsigned l = 0;
	unsigned fftLen = 1;
	while (fftLen < u_halfWidth)
	{
		l++;
		fftLen <<= 1;
	}
	float fLen = (float)fftLen;

	d_ScaleWindow(halfWinlen, u_halfWidth, s_Wnd, s_res_wnd, fLen);

	for (unsigned i = 1 + workerId; i < fftLen; i += numWorker)
	{
		s_res_wnd[i] += s_res_wnd[i+fftLen];
		s_res_wnd[i + fftLen] = 0.0f;
	}
	__syncthreads();
	
	d_fft(s_res_wnd, l);

	float rate = halfWinlen / fLen;
	for (unsigned i = workerId; i < fftLen; i += numWorker)
	{
		float v = 0.0f;
		if (i < uSpecLen)
		{
			float amplitude = sqrtf(s_res_wnd[i] * s_res_wnd[i] + s_res_wnd[i + fftLen] * s_res_wnd[i + fftLen]);
			v = amplitude*rate;
		}
		s_res_wnd[i] = v;
	}

	__syncthreads();	

	for (unsigned i = fftLen+workerId; i < 2*fftLen; i += numWorker)
	{
		s_res_wnd[i] = 0.0f;
	}
	__syncthreads();
}


inline __device__ float d_SymWndGetSample(const float* s_SymWnd, unsigned u_halfWidth, int i)
{
	if (i< -(int)(u_halfWidth - 1) || i>u_halfWidth - 1) return 0.0f;
	int idst = i >= 0 ? i : -i;
	float v = s_SymWnd[idst];
	return i >= 0 ? v : -v;
}

__device__ void d_ScaleSymWindow(float srcHalfWinlen, unsigned u_SrcHalfWidth, const float* s_SymWnd, float* s_dstSymWnd, float targetHalfWidth)
{
	unsigned u_TargetHalfWidth = (unsigned)ceilf(targetHalfWidth);
	float rate = srcHalfWinlen / targetHalfWidth;
	bool interpolation = rate < 1.0f;

	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	for (int i = workerId; i <= (int)(u_TargetHalfWidth - 1); i += numWorker)
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

			float p0 = d_SymWndGetSample(s_SymWnd, u_SrcHalfWidth, ipos0);
			float p1 = d_SymWndGetSample(s_SymWnd, u_SrcHalfWidth, ipos1);
			float p2 = d_SymWndGetSample(s_SymWnd, u_SrcHalfWidth, ipos2);
			float p3 = d_SymWndGetSample(s_SymWnd, u_SrcHalfWidth, ipos3);

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
				sum += d_SymWndGetSample(s_SymWnd, u_SrcHalfWidth, ipos);
			}
			destValue = sum / (float)(ipos2 - ipos1 + 1);
		}
		s_dstSymWnd[i] = destValue;
	}
	__syncthreads();
}

__device__ void d_CreateSymmetricWindowFromAmpSpec(float* s_spectrum, unsigned uSpecLen, float halfWinlen, unsigned u_halfWidth, float* s_SymWnd)
{
	unsigned l = 0;
	unsigned fftLen = 1;
	while ((float)fftLen < u_halfWidth)
	{
		l++;
		fftLen <<= 1;
	}

	float rate = (float)fftLen / halfWinlen;

	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	if (workerId == 0)
	{
		s_SymWnd[0] = 0.0f;
		s_SymWnd[fftLen] = 0.0f;

		s_SymWnd[fftLen/2] = 0.0f;
		s_SymWnd[fftLen + fftLen/2] = 0.0f;
	}
	for (unsigned i = 1 + workerId; i < fftLen / 2; i += numWorker)
	{
		float x = 0.0f;
		if (i < uSpecLen)
		{
			float amplitude = s_spectrum[i];
			x = amplitude * rate;
		}
		s_SymWnd[i] = 0.0f;
		s_SymWnd[fftLen + i] = x;
		s_SymWnd[fftLen - i] = 0.0f;
		s_SymWnd[2 * fftLen - i] = -x;
	}
	__syncthreads();

	d_ifft(s_SymWnd, l);

	for (unsigned i = workerId; i < fftLen; i += numWorker)
	{
		float window = (cosf((float)i * (float)PI / (float)fftLen) + 1.0f)*0.5f;
		s_SymWnd[i + fftLen] = s_SymWnd[i]*window;
	}
	__syncthreads();

	d_ScaleSymWindow((float)fftLen, fftLen, s_SymWnd + fftLen, s_SymWnd, halfWinlen);

}

__device__ void SymWin_Repitch_FormantPreserved(float srcHalfWinLen, float* srcBuf, float dstHalfWinLen, float* dstBuf, float k)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	unsigned u_TargetHalfWidth = (unsigned)ceilf(dstHalfWinLen);
	unsigned uSrcHalfWidth = (unsigned)ceilf(srcHalfWinLen);

	float rate = dstHalfWinLen / srcHalfWinLen;
	float amplitude = k*sqrtf(rate);
	for (unsigned i = workerId; (float)i < dstHalfWinLen; i += numWorker)
	{
		float dstV = 0.0f;
		float srcPos = (float)i;
		unsigned uSrcPos = (unsigned)(srcPos + 0.5f);
		while (uSrcPos < uSrcHalfWidth)
		{
			dstV += srcBuf[uSrcPos];
			srcPos += dstHalfWinLen;
			uSrcPos = (unsigned)(srcPos + 0.5f);
		}
		srcPos = dstHalfWinLen - (float)i;
		uSrcPos = (unsigned)(srcPos + 0.5f);
		while (uSrcPos < uSrcHalfWidth)
		{
			dstV -= srcBuf[uSrcPos];
			srcPos += dstHalfWinLen;
			uSrcPos = (unsigned)(srcPos + 0.5f);
		}
		float window = (cosf((float)i * (float)PI / dstHalfWinLen) + 1.0f)*0.5f;
		dstBuf[i] += dstV*amplitude*window;
	}
	__syncthreads();
}

__device__ void SymWin_WriteToBuf(unsigned dstLen, float* d_dstBuf, float pos, float halfWinLen, float* s_symWnd)
{
	int ipos = (int)floorf(pos);
	unsigned uHalfWinLen = (unsigned)ceilf(halfWinLen);
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	for (int i = (int)(workerId)-(int)(uHalfWinLen - 1); i <= (int)(uHalfWinLen - 1); i += (int)numWorker)
	{
		int uI = i < 0 ? -i : i;
		float v = s_symWnd[uI];
		int dstI = ipos + i;
		if (dstI >= 0 && dstI < dstLen)
			d_dstBuf[dstI] += i<0? -v:v;
	}
	__syncthreads();
}

__device__ void Win_WriteToBuf(unsigned dstLen, float* d_dstBuf, float pos, float halfWinLen, float* s_Wnd)
{
	int ipos = (int)floorf(pos);
	unsigned uHalfWinLen = (unsigned)ceilf(halfWinLen);
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	for (int i = (int)(workerId)-(int)(uHalfWinLen - 1); i <= (int)(uHalfWinLen - 1); i += (int)numWorker)
	{
		int uI = i < 0 ? halfWinLen*2 + i : i;
		float v = s_Wnd[uI];
		int dstI = ipos + i;
		if (dstI >= 0 && dstI < dstLen)
			d_dstBuf[dstI] += v;
	}
	__syncthreads();
}

__device__ void AmpSpec_Scale(float srcHalfWinLen, float* srcBuf, float dstHalfWinLen, float* dstBuf, float k)
{
	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	unsigned srcSpecLen = (unsigned)ceilf(srcHalfWinLen*0.5f);
	unsigned specLen = (unsigned)ceilf(dstHalfWinLen*0.5f);
	float rate = srcHalfWinLen / dstHalfWinLen;
	float mulRate = sqrtf(dstHalfWinLen / srcHalfWinLen) *k;
	bool interpolation = rate < 1.0f;

	for (unsigned i = workerId; i < specLen; i += numWorker)
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

			float p0 = ipos0 < 0 ? 0.0f : srcBuf[ipos0];
			float p1 = ipos1 >= srcSpecLen ? 0.0f : srcBuf[ipos1];
			float p2 = ipos2 >= srcSpecLen ? 0.0f : srcBuf[ipos2];
			float p3 = ipos3 >= srcSpecLen ? 0.0f : srcBuf[ipos3];

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
				sum += (ipos<0 || ipos >= srcSpecLen) ? 0.0f : srcBuf[ipos];
			}
			destValue = sum / (float)(ipos2 - ipos1 + 1);

		}

		dstBuf[i] += destValue*mulRate;

	}
	__syncthreads();

}


__device__ void d_CreateNoiseWindowFromAmpSpec(float* s_spectrum, float* randphase, unsigned uSpecLen, float halfWinlen, unsigned u_halfWidth, float* s_NoiseWnd, float targetHalfWidth)
{
	unsigned l = 0;
	unsigned fftLen = 1;
	while ((float)fftLen < u_halfWidth)
	{
		l++;
		fftLen <<= 1;
	}
	float rate = (float)fftLen / halfWinlen;

	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	if (workerId == 0)
	{
		s_NoiseWnd[0] = 0.0f;
		s_NoiseWnd[fftLen] = 0.0f;

		s_NoiseWnd[fftLen / 2] = 0.0f;
		s_NoiseWnd[fftLen + fftLen / 2] = 0.0f;
	}
	for (unsigned i = 1 + workerId; i < fftLen / 2; i += numWorker)
	{
		float x = 0.0f;
		float y = 0.0f;
		if (i < uSpecLen)
		{
			float amplitude = s_spectrum[i] * rate;
			float phase = randphase[i] * 2.0f*PI;
			x = amplitude * cosf(phase);
			y = amplitude * sinf(phase);
		}
		s_NoiseWnd[i] = x;
		s_NoiseWnd[fftLen + i] = y;
		s_NoiseWnd[fftLen - i] = x;
		s_NoiseWnd[2 * fftLen - i] = -y;
	}
	__syncthreads();

	d_ifft(s_NoiseWnd, l);

	unsigned u_targetHalfWidth = (unsigned)ceilf(targetHalfWidth);
	unsigned skip_len = max(2 * fftLen, 2 * u_targetHalfWidth);

	for (unsigned i = workerId; i < fftLen; i += numWorker)
	{
		float window = (cosf((float)i * (float)PI / (float)fftLen) + 1.0f)*0.5f;

		s_NoiseWnd[i + skip_len] = s_NoiseWnd[i] * window;
		if (i>0)
			s_NoiseWnd[skip_len+ 2 * fftLen - i] = s_NoiseWnd[fftLen - i] * window;
	}
	if (workerId == 0)
		s_NoiseWnd[skip_len + fftLen] = 0;
	__syncthreads();

	d_ScaleWindow((float)fftLen, fftLen, s_NoiseWnd + skip_len, s_NoiseWnd, targetHalfWidth);
}

