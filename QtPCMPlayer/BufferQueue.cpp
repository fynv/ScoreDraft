#include "BufferQueue.h"

void BufferQueue::AddBuffer(AudioBuffer_Deferred buf)
{
	unsigned size = (unsigned)buf->size();
	m_queue.push(buf);
	m_totalBufferLenth += size;
}

void BufferQueue::SetCursor(unsigned pos)
{
	m_GetPos_rel = (int)pos - (int)m_HeadPos;
}

unsigned BufferQueue::GetCursor()
{
	return (unsigned)((int)m_HeadPos + m_GetPos_rel);
}

short BufferQueue::GetSample()
{
	if (m_GetPos_rel < 0)
	{
		m_GetPos_rel++;
		return 0;
	}
	while (!m_queue.empty())
	{
		AudioBuffer_Deferred buf = m_queue.front();
		unsigned size = (unsigned)buf->size();
		if ((unsigned)m_GetPos_rel < size)
		{
			short value = (*buf)[m_GetPos_rel];
			m_GetPos_rel++;
			return value;
		}
		m_queue.pop();
		m_HeadPos += size;
		m_GetPos_rel -= size;
		m_totalBufferLenth -= size;
	}
	return 0;
}

unsigned BufferQueue::GetRemainingSamples()
{
	if ((int)m_totalBufferLenth > m_GetPos_rel) return m_totalBufferLenth - m_GetPos_rel;
	else return 0;
}


