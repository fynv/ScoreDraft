#include "helper_math.h"
#define PI ((float)3.1415926535897932384626433832795)

__device__
unsigned short reverseBits(unsigned short x, unsigned l)
{
	x = (((x & 0xaaaa) >> 1) | ((x & 0x5555) << 1));
	x = (((x & 0xcccc) >> 2) | ((x & 0x3333) << 2));
	x = (((x & 0xf0f0) >> 4) | ((x & 0x0f0f) << 4));
	x = ((x >> 8) | (x << 8));
	return x >> (16 - l);
}

__device__ float2 CompMul(const float2& c1, const float2& c2)
{
	float2 ret;
	ret.x = c1.x*c2.x - c1.y*c2.y;
	ret.y = c1.x*c2.y + c1.y*c2.x;
	return ret;
}

__device__ float2 CompPowN(float2 c, unsigned n)
{
	float2 mul = c;
	float2 ret = make_float2(1.0f, 0.0f);

	while (n > 0)
	{
		if (n & 1)
		{
			ret = CompMul(ret, mul);
		}
		mul = CompMul(mul, mul);		
		n >>= 1;
	}
	return ret;	
}

__device__
void d_fft(float* a, unsigned l)
{
	unsigned n = 1 << l;

	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	for (unsigned short i = (unsigned short)workerId+1; i< (unsigned short)n - 1; i += (unsigned short)numWorker)
	{
		unsigned short j = reverseBits(i,l);
		unsigned u, v;
		if (i < j)
		{
			u = i;
			v = j;
		}
		else if (j<i)
		{
			u = i + n;
			v = j + n;
		}
		else continue;
		float t = a[u];
		a[u] = a[v];
		a[v] = t;
	}

	__syncthreads();

	unsigned le = 1;
	for (unsigned m = 1; m <= l; m++)
	{
		unsigned lei = le;
		le <<= 1;

		float tmp = PI / (float)lei;
		float2 w;
		w.x = cosf(tmp); w.y = -sinf(tmp);

		unsigned gsize = n / le;
		unsigned workPerWorker = (n / 2 - 1) / numWorker + 1;
		unsigned lastj = workerId*workPerWorker/gsize;
		float2 u = CompPowN(w, lastj);

		for (unsigned work = workerId*workPerWorker; work < (workerId + 1)*workPerWorker && work<n/2; work++)
		{
			unsigned j = work / gsize;
			unsigned i = (work % gsize)*le + j;
			unsigned ip = i + lei;

			if (j > lastj)
			{
				u = CompMul(u, w);
				lastj = j;
			}

			float2 comp_a_i = make_float2(a[i], a[i + n]);
			float2 comp_a_ip = make_float2(a[ip], a[ip + n]);
			float2 t = CompMul(u, comp_a_ip);
			comp_a_ip = comp_a_i - t;
			comp_a_i = comp_a_i + t;
			a[i] = comp_a_i.x; a[i + n] = comp_a_i.y;
			a[ip] = comp_a_ip.x; a[ip + n] = comp_a_ip.y;
		}
		__syncthreads();
	}
}

__global__
void g_fft_test(float* d_buf, unsigned l)
{
	d_fft(d_buf, l);
}

void h_fft_test(float* d_buf, unsigned l)
{
	g_fft_test << <1, 256 >> >(d_buf, l);
}


__device__
void d_ifft(float* a, unsigned l)
{
	unsigned n = 1 << l;

	unsigned numWorker = blockDim.x;
	unsigned workerId = threadIdx.x;

	for (unsigned short i = (unsigned short)workerId + 1; i< (unsigned short)n - 1; i += (unsigned short)numWorker)
	{
		unsigned short j = reverseBits(i, l);
		unsigned u, v;
		if (i < j)
		{
			u = i;
			v = j;
		}
		else if (j<i)
		{
			u = i + n;
			v = j + n;
		}
		else continue;
		float t = a[u];
		a[u] = a[v];
		a[v] = t;
	}

	__syncthreads();

	unsigned le = 1;
	for (unsigned m = 1; m <= l; m++)
	{
		unsigned lei = le;
		le <<= 1;

		float tmp = PI / (float)lei;
		float2 w;
		w.x = cosf(tmp); w.y = sinf(tmp);

		unsigned gsize = n / le;
		unsigned workPerWorker = (n / 2 - 1) / numWorker + 1;
		unsigned lastj = workerId*workPerWorker / gsize;
		float2 u = CompPowN(w, lastj);
		u.x *= 0.5f; u.y *= 0.5f;

		for (unsigned work = workerId*workPerWorker; work < (workerId + 1)*workPerWorker && work<n / 2; work++)
		{
			unsigned j = work / gsize;
			unsigned i = (work % gsize)*le + j;
			unsigned ip = i + lei;

			if (j > lastj)
			{
				u = CompMul(u, w);
				lastj = j;
			}

			float2 comp_a_i = make_float2(0.5f*a[i], 0.5f*a[i + n]);
			float2 comp_a_ip = make_float2(a[ip], a[ip + n]);
			float2 t = CompMul(u, comp_a_ip);
			comp_a_ip = comp_a_i - t;
			comp_a_i = comp_a_i + t;
			a[i] = comp_a_i.x; a[i + n] = comp_a_i.y;
			a[ip] = comp_a_ip.x; a[ip + n] = comp_a_ip.y;
		}
		__syncthreads();


	}

}


__global__
void g_ifft_test(float* d_buf, unsigned l)
{
	d_ifft(d_buf, l);
}

void h_ifft_test(float* d_buf, unsigned l)
{
	g_ifft_test << <1, 256 >> >(d_buf, l);
}
