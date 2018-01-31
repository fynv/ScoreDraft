#include "BufferQueue.h"

void BufferQueue::AddBuffer(AudioBuffer_Deferred buf)
{
	unsigned size = (unsigned)buf->size();
	unsigned alignPos = buf->m_AlignPos;
	m_queue.push_back(buf);
	m_totalBufferLenth += size - alignPos;
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
		unsigned alignPos = buf->m_AlignPos;
		if ((unsigned)m_GetPos_rel < size - alignPos)
		{
			short value = (*buf)[alignPos+(unsigned)m_GetPos_rel];

			std::list<AudioBuffer_Deferred>::iterator iter = m_queue.begin();
			iter++;
			if (iter != m_queue.end())
			{
				AudioBuffer_Deferred buf_next = *iter;
				if ((unsigned)m_GetPos_rel >= size - alignPos - buf_next->m_AlignPos)
				{
					short value2 = (*buf_next)[(unsigned)m_GetPos_rel - (size - alignPos - buf_next->m_AlignPos)];
					int v= (int)value + (int)value2;
					if (v > 32767) v = 32767;
					else if (v < -32767) v = -32767;
					value = (short)v;
				}

			}

			m_GetPos_rel++;
			return value;
		}
		m_queue.pop_front();
		m_HeadPos += size - alignPos;
		m_GetPos_rel -= size - alignPos;
		m_totalBufferLenth -= size - alignPos;
	}
	return 0;
}

unsigned BufferQueue::GetRemainingSamples()
{
	if ((int)m_totalBufferLenth > m_GetPos_rel) return m_totalBufferLenth - m_GetPos_rel;
	else return 0;
}


