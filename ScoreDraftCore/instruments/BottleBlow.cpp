#include "instruments/BottleBlow.h"
#include "Note.h"
#include "TrackBuffer.h"

#include <cmath>
#include <time.h>

#define PI 3.14159265359f

BottleBlow::BottleBlow()
{
	srand((unsigned)time(NULL));
	m_accelerate = false;
}

BottleBlow::~BottleBlow()
{
}

inline float rand01()
{
	float f= (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}

inline float randGauss()
{
	return 100.0f*sqrtf(-log(rand01()))*cos(rand01()*PI);
}

void BottleBlow::GenerateNoteWave(unsigned numOfSamples, float sampleFreq, NoteBuffer* noteBuf)
{
	noteBuf->m_sampleNum = numOfSamples;
	noteBuf->Allocate();

	unsigned j;
	float out = 0.0f;
	float Dout = 0.0f;

	//float FreqCut = 1.0f / 5000.0f;
	float k = 0.02f;
	float FreqCut = k*sampleFreq;
	float a = powf(2 * PI, 2.0f)*sqrtf(powf(FreqCut, 4.0f) + powf(sampleFreq, 4.0f));
	//float b = 2 * PI * powf(2.0f*(sqrtf(powf(FreqCut, 4.0f) + powf(sampleFreq, 4.0f)) - powf(sampleFreq, 2.0f)),0.5f);
	float b = 2 * PI * FreqCut*FreqCut / sampleFreq;

	float ampfac = powf(FreqCut, 1.5f);

	for (j = 0; j < noteBuf->m_sampleNum; j++)
	{
		float x2 = ((float)j / (float)(noteBuf->m_sampleNum - 1));
		float amplitude = 1.0f - powf(x2 - 0.5f, 3.0f)*8.0f;

		noteBuf->m_data[j] = amplitude*out*ampfac;

		//float e = randGauss();
		float e = rand01() - 0.5f;
		float DDout = e - b*Dout - a*out;
		Dout += DDout;
		out += Dout;

	}
}


