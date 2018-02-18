#include "BufferQueue.h"

void BufferQueue::AddBuffer(AudioBuffer_Deferred buf)
{
	unsigned size = (unsigned)buf->Size();
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

void BufferQueue::GetSample(short sample[2])
{
	if (m_GetPos_rel < 0)
	{
		m_GetPos_rel++;
		sample[0] = sample[1] = 0;
		return;
	}
	while (!m_queue.empty())
	{
		AudioBuffer_Deferred buf = m_queue.front();
		unsigned size = (unsigned)buf->Size();
		unsigned chn = buf->m_chn;
		unsigned alignPos = buf->m_AlignPos;
		if ((unsigned)m_GetPos_rel < size - alignPos)
		{
			unsigned getPos = alignPos + (unsigned)m_GetPos_rel;
			short value[2];

			if (chn == 1)
			{
				value[0] = value[1] = (*buf)[getPos];
			}
			else if (chn == 2)
			{
				value[0] = (*buf)[getPos * 2];
				value[1] = (*buf)[getPos * 2 +1];
			}
			
			std::list<AudioBuffer_Deferred>::iterator iter = m_queue.begin();
			iter++;
			if (iter != m_queue.end())
			{
				AudioBuffer_Deferred buf_next = *iter;
				chn = buf_next->m_chn;
				if ((unsigned)m_GetPos_rel >= size - alignPos - buf_next->m_AlignPos)
				{
					getPos = (unsigned)m_GetPos_rel - (size - alignPos - buf_next->m_AlignPos);
					short value2[2];
					if (chn == 1)
					{
						value2[0] = value2[1] = (*buf_next)[getPos];
					}
					else if (chn == 2)
					{
						value2[0] = (*buf_next)[getPos * 2];
						value2[1] = (*buf_next)[getPos * 2 + 1];
					}

					for (unsigned i = 0; i < 2; i++)
					{
						int v = (int)value[i] + (int)value2[i];
						if (v > 32767) v = 32767;
						else if (v < -32767) v = -32767;
						value[i] = (short)v;
					}
				}

			}

			m_GetPos_rel++;

			sample[0] = value[0];
			sample[1] = value[1];

			return;
		}
		m_queue.pop_front();
		m_HeadPos += size - alignPos;
		m_GetPos_rel -= size - alignPos;
		m_totalBufferLenth -= size - alignPos;
	}
	sample[0] = sample[1] = 0;
}

float BufferQueue::GetSample()
{
	short sample[2];
	GetSample(sample);
	return ((float)sample[0] + (float)sample[1]) / 65536.0f;
}

unsigned BufferQueue::GetRemainingSamples()
{
	if ((int)m_totalBufferLenth > m_GetPos_rel) return m_totalBufferLenth - m_GetPos_rel;
	else return 0;
}


