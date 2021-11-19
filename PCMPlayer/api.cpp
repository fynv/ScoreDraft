#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif


extern "C"
{
	SCOREDRAFT_API void* PCMPlayerCreate(double sample_rate, unsigned ui);
	SCOREDRAFT_API void PCMPlayerDestroy(void* ptr);
	SCOREDRAFT_API void PlayTrack(void* ptr, void* ptr_track);
	SCOREDRAFT_API float GetRemainingTime(void* ptr);
	SCOREDRAFT_API void MainLoop(void* ptr);
}

#include "PCMPlayer.h"

void* PCMPlayerCreate(double sample_rate, unsigned ui)
{
	return new PCMPlayer(sample_rate, ui!=0);
}

void PCMPlayerDestroy(void* ptr)
{
	PCMPlayer* player = (PCMPlayer*)ptr;
	delete player;
}

void PlayTrack(void* ptr, void* ptr_track)
{
	PCMPlayer* player = (PCMPlayer*)ptr;
	TrackBuffer* track = (TrackBuffer*)ptr_track;
	player->PlayTrack(*track);
}

float GetRemainingTime(void* ptr)
{
	PCMPlayer* player = (PCMPlayer*)ptr;
	return player->GetRemainingTime();
}

void MainLoop(void* ptr)
{
	PCMPlayer* player = (PCMPlayer*)ptr;
	player->main_loop();
}
