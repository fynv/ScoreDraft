#ifndef _FrqData_h
#define _FrqData_h

#include <vector>

struct FrqDataPoint
{
	double freq;
	double dyn;
};

class FrqData : public std::vector<FrqDataPoint>
{
public:
	int m_window_interval;
	double m_key_freq;

	bool ReadFromFile(const char* filename);
};


#endif
