#ifndef _WavIO_h
#define _WavIO_h

#include "TrackBuffer.h"

void WriteToWav(TrackBuffer& track, const char* fileName);
void ReadFromWav(TrackBuffer& track, const char* fileName);


#endif