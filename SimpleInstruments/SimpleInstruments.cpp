#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	SCOREDRAFT_API void GeneratePureSin(void* ptr_wavbuf, float freq, float fduration);
	SCOREDRAFT_API void GenerateSquare(void* ptr_wavbuf, float freq, float fduration);
	SCOREDRAFT_API void GenerateTriangle(void* ptr_wavbuf, float freq, float fduration);
	SCOREDRAFT_API void GenerateSawtooth(void* ptr_wavbuf, float freq, float fduration);
	SCOREDRAFT_API void GenerateNaivePiano(void* ptr_wavbuf, float freq, float fduration);
	SCOREDRAFT_API void GenerateBottleBlow(void* ptr_wavbuf, float freq, float fduration);
}

#include <WavBuffer.h>
#include <cmath>

#define PI 3.14159265359f

void GeneratePureSin(void* ptr_wavbuf, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(1, len);

	float wave = 1.0f;
	float Dwave = 0.0f;
	float a = powf(2.0f * PI*sampleFreq, 2.0f);

	for (size_t j = 0; j < len; j++)
	{
		float amplitude = sinf(PI*(float)j / (float)len);
		wavbuf->m_data[j] = amplitude * wave;
		float DDwave = -a * wave;
		Dwave += DDwave;
		wave += Dwave;
	}
}

void GenerateSquare(void* ptr_wavbuf, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(1, len);

	for (size_t j = 0; j < len; j++)
	{
		float x = sampleFreq * j;
		x = x - floor(x);
		float wave = x > 0.5f ? -1.0f : 1.0f;
		wavbuf->m_data[j] = wave;
	}
}

void GenerateTriangle(void* ptr_wavbuf, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(1, len);

	for (size_t j = 0; j < len; j++)
	{
		float amplitude = 1.0f - 2.0f*fabsf((float)j / (float)(len - 1) - 0.5f);
		float x = sampleFreq * j;
		x = x - floor(x);
		float wave = x > 0.5f ? (x - 0.75f)*4.0f : (0.25f - x)*4.0f;
		wavbuf->m_data[j] = wave * amplitude;
	}
}

void GenerateSawtooth(void* ptr_wavbuf, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(1, len);

	for (size_t j = 0; j < len; j++)
	{
		float amplitude = 1.0f - ((float)j / (float)(len - 1));
		float phase = sampleFreq * j;
		float wave = 1.0f - 2.0f*(phase - floor(phase));
		wavbuf->m_data[j] = amplitude * wave;
	}
}

void GenerateNaivePiano(void* ptr_wavbuf, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(1, len);
	
	for (size_t j = 0; j < len; j++)
	{
		float x = sampleFreq * j;
		x = x - floor(x);

		float x2 = (float)j / fNumOfSamples;

		float amplitude = 1.0f - powf(x2 - 0.5f, 3.0f)*8.0f;
		float wave = (1.0f + 0.5f*cos(2 * PI*x * 5))*sin(PI*x)* powf(1.0f - 2.0f * x, 3.0f);

		wavbuf->m_data[j] = amplitude * wave;
	}
}

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}


void GenerateBottleBlow(void* ptr_wavbuf, float freq, float fduration)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	float sampleRate = wavbuf->m_sampleRate;

	float fNumOfSamples = fduration * sampleRate*0.001f;
	size_t len = (size_t)ceilf(fNumOfSamples);
	float sampleFreq = freq / sampleRate;

	wavbuf->Allocate(1, len);	

	float out = 0.0f;
	float Dout = 0.0f;

	//float FreqCut = 1.0f / 5000.0f;
	float k = 0.02f;
	float FreqCut = k * sampleFreq;
	float a = powf(2 * PI, 2.0f)*sqrtf(powf(FreqCut, 4.0f) + powf(sampleFreq, 4.0f));
	//float b = 2 * PI * powf(2.0f*(sqrtf(powf(FreqCut, 4.0f) + powf(sampleFreq, 4.0f)) - powf(sampleFreq, 2.0f)),0.5f);
	float b = 2 * PI * FreqCut*FreqCut / sampleFreq;

	float ampfac = powf(FreqCut, 1.5f);

	for (size_t j = 0; j < len; j++)
	{
		float x2 = (float)j / fNumOfSamples;
		float amplitude = 1.0f - powf(x2 - 0.5f, 3.0f)*8.0f;

		wavbuf->m_data[j] = amplitude * out*ampfac;

		//float e = randGauss();
		float e = rand01() - 0.5f;
		float DDout = e - b * Dout - a * out;
		Dout += DDout;
		out += Dout;
	}
}
