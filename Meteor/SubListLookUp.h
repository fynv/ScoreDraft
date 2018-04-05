#ifndef _SubListLookUp_h
#define _SubListLookUp_h

#include <vector>
#include <float.h>
#include <stdio.h>

typedef std::vector<unsigned> SubList;

class SubLists
{
public:
	float m_minStart;
	float m_maxEnd;
	float m_interval;
	std::vector<SubList> m_subLists;

	SubLists()	{}

	unsigned GetIntervalId(float v) const
	{
		if (v < m_minStart) return 0;
		unsigned id = (unsigned)((v - m_minStart) / m_interval);
		if (id >= (unsigned)m_subLists.size()) id = (unsigned)m_subLists.size() - 1;
		return id;
	}

	void SaveToFile(FILE* fp) const
	{
		fwrite(&m_minStart, sizeof(float), 3, fp);
		unsigned count = (unsigned)m_subLists.size();
		fwrite(&count, sizeof(unsigned), 1, fp);
		for (unsigned i = 0; i < count; i++)
		{
			unsigned sub_count = (unsigned)m_subLists[i].size();
			fwrite(&sub_count, sizeof(unsigned), 1, fp);
			fwrite(&m_subLists[i][0], sizeof(unsigned), sub_count, fp);
		}
	}

	void LoadFromFile(FILE* fp)
	{
		fread(&m_minStart, sizeof(float), 3, fp);
		unsigned count;
		fread(&count, sizeof(unsigned), 1, fp);
		m_subLists.clear();
		m_subLists.resize(count);
		for (unsigned i = 0; i < count; i++)
		{
			unsigned sub_count;
			fread(&sub_count, sizeof(unsigned), 1, fp);
			m_subLists[i].resize(sub_count);
			fread(&m_subLists[i][0], sizeof(unsigned), sub_count, fp);
		}
	}

	template<class T>
	void SetData(const std::vector<T>& fullList, float interval)
	{
		m_interval = interval;
		m_minStart = FLT_MAX;
		m_maxEnd = -FLT_MAX;
		if (fullList.size()==0) return;
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
			unsigned startInterval = (unsigned)((fullList[i].start - m_minStart) / interval);
			unsigned endInterval = (unsigned)((fullList[i].end - m_minStart) / interval);
			if (endInterval >= numIntervals) endInterval = numIntervals - 1;

			for (unsigned j = startInterval; j <= endInterval; j++)
			{
				m_subLists[j].push_back(i);
			}

		}

	}

};

#endif
