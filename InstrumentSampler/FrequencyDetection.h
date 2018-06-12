#ifndef _FrequencyDetection_h
#define _FrequencyDetection_h

#include <memory.h>
#include <math.h>
#include <vector>

class Buffer
{
	std::vector<float> _data;
public:
	unsigned m_size;
	float* m_data;

	void Allocate(unsigned size)
	{
		_data.resize(size);
		m_size = size;
		m_data = &_data[0];
		SetZero();
	}

	float GetSample(int i) const
	{
		if (i<0 || i >= (int)m_size) return 0.0f;
		return m_data[i];
	}

	void SetZero()
	{
		memset(m_data, 0, sizeof(float)*m_size);
	}

	void SetSample(int i, float v)
	{
		if (i < 0 || i >= (int)m_size) return;
		m_data[i] = v;
	}

	void AddToSample(int i, float v)
	{
		if (i < 0 || i >= (int)m_size) return;
		m_data[i] += v;
	}

	float GetMax()
	{
		float maxv = 0.0f;
		for (size_t i = 0; i < m_size; i++)
		{
			if (fabsf(m_data[i])>maxv) maxv = fabsf(m_data[i]);
		}
		return maxv;
	}

};

float fetchFrequency(const Buffer& buf, unsigned sampleRate);

#endif
