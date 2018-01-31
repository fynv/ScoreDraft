#ifndef _BufferQueue_h
#define _BufferQueue_h

#include <Deferred.h>
#include <vector>
#include <list>

class AudioBuffer : public std::vector<short>
{
public:
	AudioBuffer()
	{
		m_AlignPos = 0;
	}
	unsigned m_AlignPos;
};

typedef Deferred<AudioBuffer> AudioBuffer_Deferred;

class BufferQueue
{
public:
	BufferQueue()
	{
		m_HeadPos = 0;
		m_GetPos_rel = 0;
		m_totalBufferLenth = 0;
	}

	void AddBuffer(AudioBuffer_Deferred buf);
	void SetCursor(unsigned pos);
	unsigned GetCursor();
	short GetSample();
	unsigned GetRemainingSamples();

private:
	std::list<AudioBuffer_Deferred> m_queue;
	unsigned m_HeadPos;
	int m_GetPos_rel;
	unsigned m_totalBufferLenth;

};


#endif

