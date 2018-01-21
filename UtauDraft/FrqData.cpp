#include "FrqData.h"
#include <stdio.h>

bool FrqData::ReadFromFile(const char* filename)
{
	FILE* fp = fopen(filename, "rb");
	if (!fp) return false;

	fseek(fp, 8, SEEK_SET);
	fread(&m_window_interval, sizeof(int), 1, fp);

	fseek(fp, 12, SEEK_SET);
	fread(&m_key_freq, sizeof(double), 1, fp);

	int count;
	fseek(fp, 36, SEEK_SET);
	fread(&count, sizeof(int), 1, fp);

	this->resize((size_t)count);

	fseek(fp, 40, SEEK_SET);
	fread(this->data(), sizeof(FrqDataPoint), count, fp);

	fclose(fp);

	return true;
}
