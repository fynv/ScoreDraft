#include "PrefixMap.h"

#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cmath>

bool PrefixMap::LoadFromFile(const char* filename)
{
	FILE *fp = fopen(filename, "r");
	if (!fp) return false;

	char line[2048];
	while (!feof(fp) && fgets(line, 2048, fp))
	{
		char pitch[20];
		char prefix[20];
		sscanf(line, "%s %s", pitch, prefix);
		(*this)[pitch]=prefix;
	}

	return true;
}

std::string PrefixMap::GetPrefixFromFreq(float freq)
{
	if (this->size() == 0) return "";
	static float centerC = 440.0f * powf(2.0f, - 9.0f / 12.0f); // C4
	static float lowest = centerC* powf(2.0f, -3.0f); // C1
	static float highest = centerC*powf(2.0, 3.0f + 11.0f / 12.0f); // B7

	if (freq < lowest) return (*this)["C1"];
	else if (freq>highest) return (*this)["B7"];

	static const char* nameMap[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };

	float fPitch = logf(freq / centerC) / logf(2.0f) + 4.0f;
	unsigned octave = (unsigned)(fPitch);
	unsigned pitchInOctave = (unsigned)( (fPitch - (float)octave)*12.0f);
	if (pitchInOctave == 12)
	{
		pitchInOctave = 0;
		octave++;
	}
	
	std::string name = std::string(nameMap[pitchInOctave]) + std::to_string(octave);
	return (*this)[name];
}
