#ifndef _SubListLookUp_h
#define _SubListLookUp_h

#include <vector>
#include <float.h>

template<class T>
class SubList : public std::vector<const T*>
{
public:
	float m_start;
	float m_end;
};

template<class T>
class SubLists
{
public:
	float m_minStart;
	float m_maxEnd;
	float m_interval;
	std::vector<SubList<T>> m_subLists;

	SubLists()	{}

	unsigned GetIntervalId(float v)
	{
		if (v < m_minStart) return 0;
		unsigned id = (unsigned)((v - m_minStart) / m_interval);
		if (id >= (unsigned)m_subLists.size()) id = (unsigned)m_subLists.size() - 1;
		return id;
	}

	void SetData(const std::vector<T>& fullList, float interval)
	{
		m_interval = interval;
		m_minStart = FLT_MAX;
		m_maxEnd = -FLT_MAX;
		for (unsigned i = 0; i < (unsigned) fullList.size(); i++)
		{
			if (fullList[i].start < m_minStart) m_minStart = fullList[i].start;
			if (fullList[i].end> m_maxEnd) m_maxEnd = fullList[i].end;
		}

		unsigned numIntervals = (unsigned)ceilf((m_maxEnd - m_minStart) / interval);
		m_subLists.clear();
		m_subLists.resize(numIntervals);

		for (unsigned i = 0; i < (unsigned)fullList.size(); i++)
		{
			unsigned startInterval = (fullList[i].start - m_minStart) / interval;
			unsigned endInterval = (fullList[i].end - m_minStart) / interval;
			if (endInterval >= numIntervals) endInterval = numIntervals - 1;

			for (unsigned j = startInterval; j <= endInterval; j++)
			{
				m_subLists[j].push_back(&fullList[i]);
			}

		}

	}

};

#endif
