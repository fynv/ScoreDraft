#ifndef _BufferQueue_h
#define _BufferQueue_h

#include <Deferred.h>
#include <vector>
#include <queue>

typedef std::vector<short> AudioBuffer;
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
	std::queue<AudioBuffer_Deferred> m_queue;
	unsigned m_HeadPos;
	int m_GetPos_rel;
	unsigned m_totalBufferLenth;

};


#endif

