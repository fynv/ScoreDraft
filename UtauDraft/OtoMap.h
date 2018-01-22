#ifndef _OtoMap_h
#define _OtoMap_h

#include <string>
#include <map>

struct VoiceLocation
{
	std::string filename;
	float offset;
	float consonant;
	float cutoff;
	float preutterance;
	float overlap;
};

class OtoMap : public std::map<std::string, VoiceLocation>
{
public:
	bool AddOtoINIPath(const char* path);


};

#endif
