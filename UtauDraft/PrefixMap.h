#ifndef _Prefix_Map_h
#define _Prefix_Map_h

#include <map>
#include <string>

class PrefixMap : public std::map<std::string, std::string>
{
public:
	bool LoadFromFile(const char* filename);
	std::string GetPrefixFromFreq(float freq);
};

#endif
