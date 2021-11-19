#pragma once
#include <cstdint>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <vector>


inline uint64_t time_micro_sec()
{
	std::chrono::time_point<std::chrono::system_clock> tpSys = std::chrono::system_clock::now();
	std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> tpMicro
		= std::chrono::time_point_cast<std::chrono::microseconds>(tpSys);
	return tpMicro.time_since_epoch().count();
}

inline uint64_t time_milli_sec()
{
	return (time_micro_sec() + 500) / 1000;
}

inline double time_sec()
{
	return (double)time_micro_sec() / 1000000.0;
}



#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


typedef std::vector<void*> PtrArray;
typedef std::vector<float> F32Buf;

inline void CalcPan(float pan, float& l, float& r)
{
	if (pan == 0.0f) return;
	else if (pan < 0.0f)
	{
		pan = -pan;
		float ll = l;
		float rl = r * pan;
		float rr = r * (1.0f - pan);
		l = ll + rl;
		r = rr;
	}
	else
	{
		float ll = l * (1.0f - pan);
		float lr = l * pan;
		float rr = r;
		l = ll;
		r = lr + rr;
	}
}

#include "TrackBuffer.h"
#include "ReadWav.h"
#include "WriteWav.h"

inline void WriteToWav(TrackBuffer& track, const char* fileName)
{
	unsigned numSamples = track.NumberOfSamples();
	unsigned chn = track.NumberOfChannels();
	unsigned sampleRate = track.Rate();
	float volume = track.AbsoluteVolume();
	float pan = track.Pan();

	WriteWav writer;
	writer.OpenFile(fileName);
	writer.WriteHeader(sampleRate, numSamples, chn);

	unsigned localBufferSize = track.GetLocalBufferSize();
	float *buffer = new float[localBufferSize*chn];
	unsigned pos = 0;
	while (numSamples > 0)
	{
		unsigned writeCount = min(numSamples, localBufferSize);
		track.GetSamples(pos, writeCount, buffer);
		writer.WriteSamples(buffer, writeCount, volume, pan);
		numSamples -= writeCount;
		pos += writeCount;
	}

	delete[] buffer;
}

inline void ReadFromWav(TrackBuffer& track, const char* fileName)
{
	unsigned numSamples;
	unsigned chn;
	unsigned sampleRate;

	ReadWav reader;
	reader.OpenFile(fileName);
	reader.ReadHeader(sampleRate, numSamples, chn);

	unsigned localBufferSize = track.GetLocalBufferSize();

	WavBuffer buf;
	buf.m_sampleRate = (float)sampleRate;
	buf.Allocate(chn, localBufferSize);

	while (numSamples > 0)
	{
		unsigned readCount = min(numSamples, localBufferSize);
		float maxv;
		reader.ReadSamples(buf.m_data, readCount, maxv);
		buf.m_sampleNum = readCount;
		track.WriteBlend(buf);
		track.MoveCursor((float)readCount / (float)track.Rate()*1000.0f);
		numSamples -= readCount;
	}
}

class Semaphore {
public:
	Semaphore(int count_ = 0)
		: count(count_) {}

	inline void notify()
	{
		std::unique_lock<std::mutex> lock(mtx);
		count++;
		cv.notify_one();
	}

	inline void wait()
	{
		std::unique_lock<std::mutex> lock(mtx);

		while (count == 0) {
			cv.wait(lock);
		}
		count--;
	}

private:
	std::mutex mtx;
	std::condition_variable cv;
	int count;
};
