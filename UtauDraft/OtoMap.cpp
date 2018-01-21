#include "OtoMap.h"

#include <stdio.h>
#include <string.h>
#include <memory.h>

bool OtoMap::AddOtoINIPath(const char* path)
{
	char otoIniPath[2048];
	sprintf(otoIniPath, "%s/oto.ini", path);

	FILE *fp = fopen(otoIniPath, "r");
	if (!fp) return false;

	char line[2048];
	while (!feof(fp) && fgets(line, 2048, fp))
	{
		char* p = strchr(line, '=');
		if (!p) continue;

		std::string lyric;
		VoiceLocation loc;

		char fn[256];
		memcpy(fn, line, p - line);
		fn[p - line] = 0;

		loc.filename = std::string(path) + "/" + fn;
		
		char seg[128];

		// lyric
		p++;
		{
			char* p2 = strchr(p, ',');
			if (!p2) continue;
			if (p2>p) memcpy(seg, p, p2 - p);
			seg[p2 - p] = 0;
			p = p2 + 1;
		}

		if (seg[0] == 0)
		{
			memcpy(seg, fn, strlen(fn) - 4);
			seg[strlen(fn) - 4] = 0;
		}

		lyric = seg;

		// shift
		{
			char* p2 = strchr(p, ',');
			if (!p2) continue;
			if (p2>p) memcpy(seg, p, p2 - p);
			seg[p2 - p] = 0;
			p = p2 + 1;
		}
		loc.shift = (float)atof(seg);

		// consonant
		{
			char* p2 = strchr(p, ',');
			if (!p2) continue;
			if (p2>p) memcpy(seg, p, p2 - p);
			seg[p2 - p] = 0;
			p = p2 + 1;
		}
		loc.consonant = (float)atof(seg);

		// end
		{
			char* p2 = strchr(p, ',');
			if (!p2) continue;
			if (p2>p) memcpy(seg, p, p2 - p);
			seg[p2 - p] = 0;
			p = p2 + 1;
		}
		loc.end = (float)atof(seg);

		// pre
		{
			char* p2 = strchr(p, ',');
			if (!p2) continue;
			if (p2>p) memcpy(seg, p, p2 - p);
			seg[p2 - p] = 0;
			p = p2 + 1;
		}
		loc.pre = (float)atof(seg);

		// overlap
		loc.overlap = (float)atof(p);

		(*this)[lyric] = loc;
	}

	fclose(fp);

	return true;
}
