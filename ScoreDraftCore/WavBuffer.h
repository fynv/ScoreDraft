#pragma once

#include "api.h"
#include <vector>

class SCOREDRAFTCORE_API WavBuffer
{
	std::vector<float> _data;
public:
	WavBuffer() : p_data(&_data)
	{
		
	}
	void Allocate(unsigned channelNum, size_t sampleNum)
	{
		m_channelNum = channelNum;	
		m_sampleNum = sampleNum;
		p_data->resize(sampleNum * channelNum);
		m_data = &(*p_data)[0];
	}

	float m_sampleRate = 44100.0f;
	unsigned m_channelNum = 1;
	size_t m_sampleNum = 0;
	std::vector<float>* p_data;
	float* m_data = nullptr;
	unsigned m_alignPos = 0;
	float m_volume = 1.0f;
	float m_pan = 0.0f;

};


