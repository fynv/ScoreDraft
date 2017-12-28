#include "percussions/TestPerc.h"
#include "Beat.h"
#include "TrackBuffer.h"


TestPerc::TestPerc()
{
}

TestPerc::~TestPerc()
{
}

void TestPerc::GenerateBeatWave(unsigned numOfSamples, BeatBuffer* beatBuf, float BufferSampleRate)
{
	beatBuf->m_sampleNum = numOfSamples;
	beatBuf->Allocate();

	float dt = 1.0f / BufferSampleRate;

	unsigned j;

	float wave = 8000.0f;
	float Dwave = 0.0f;

	float r1 = powf(600.0f*dt, 2.0f);
	float r2 = 3.0f*dt;

	float ampFac = m_beatVolume / 8000.0f;
	
	for (j = 0; j<beatBuf->m_sampleNum; j++)
	{
		float x2 = ((float)j / (float)(beatBuf->m_sampleNum - 1));
		float amplitude = 1.0f - powf(x2 - 0.5f, 3.0f)*8.0f;

		beatBuf->m_data[j] = amplitude*wave*ampFac;

		float sign1 = wave > 0.0f ? 1.0f : -1.0f;
		float sign2 = Dwave > 0.0f ? 1.0f : -1.0f;

		float DDwave = -sign1*tan(r1*fabsf(wave)) - sign2*r2*fabsf(Dwave);

		Dwave += DDwave;
		wave += Dwave;


	}
}

