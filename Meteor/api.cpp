#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	SCOREDRAFT_API void* CtrlPntCreate(double freq, double fduration);
	SCOREDRAFT_API void CtrlPntDestroy(void* ptr);
	SCOREDRAFT_API void* SyllableCreate(const char* lyric, void* ptr_lst_ctrl_pnts);
	SCOREDRAFT_API void SyllableDestroy(void* ptr);
	SCOREDRAFT_API void EventDestroy(void* ptr);
	SCOREDRAFT_API void EventSetOffset(void* ptr, float offset);
	SCOREDRAFT_API void* EventInstCreate(unsigned instrument_id, double freq, float fduration);
	SCOREDRAFT_API void* EventPercCreate(unsigned instrument_id, float fduration);
	SCOREDRAFT_API void* EventSingCreate(unsigned instrument_id, void* ptr_syllable_list);
	SCOREDRAFT_API void* MeteorCreate0();
	SCOREDRAFT_API void* MeteorCreate(void* ptr_event_list);
	SCOREDRAFT_API void MeteorDestroy(void* ptr);
	SCOREDRAFT_API void MeteorSaveToFile(void* ptr, const char* filename);
	SCOREDRAFT_API void MeteorLoadFromFile(void* ptr, const char* filename);
	SCOREDRAFT_API void MeteorPlay(void* ptr_meteor, void* ptr_track);
	SCOREDRAFT_API void* Base64Create(void* ptr_meteor);
	SCOREDRAFT_API void Base64Destroy(void* ptr);
	SCOREDRAFT_API const char* Base64Get(void* ptr);
}

#include "Meteor.h"

void* CtrlPntCreate(double freq, double fduration)
{
	return new CtrlPnt({ freq, fduration });
}

void CtrlPntDestroy(void* ptr)
{
	delete (CtrlPnt*)ptr;
}

void* SyllableCreate(const char* lyric, void* ptr_lst_ctrl_pnts)
{
	std::vector<CtrlPnt*>* lst_ctrl_pnts = (std::vector<CtrlPnt*>*)ptr_lst_ctrl_pnts;
	Syllable* syllable = new Syllable;
	syllable->lyric = lyric;
	syllable->ctrlPnts.resize(lst_ctrl_pnts->size());
	for (size_t i = 0; i < lst_ctrl_pnts->size(); i++)
	{
		syllable->ctrlPnts[i] = *(*lst_ctrl_pnts)[i];
	}
	return syllable;
}

void SyllableDestroy(void* ptr)
{
	delete (Syllable*)ptr;
}

void EventDestroy(void* ptr)
{
	delete (Event*)ptr;
}

void EventSetOffset(void* ptr, float offset)
{
	Event* e = (Event*)ptr;
	e->offset = offset;
}


void* EventInstCreate(unsigned instrument_id, double freq, float fduration)
{
	return new EventInst(instrument_id, freq, fduration);
}

void* EventPercCreate(unsigned instrument_id, float fduration)
{
	return new EventPerc(instrument_id, fduration);
}

void* EventSingCreate(unsigned instrument_id, void* ptr_syllable_list)
{
	std::vector<const Syllable*>* syllable_list = (std::vector<const Syllable*>*)ptr_syllable_list;
	return new EventSing(instrument_id, syllable_list->size(), syllable_list->data());
}

#include "DrawText.h"

void* MeteorCreate0()
{
	Text text("测试");
	return new Meteor;
}

void* MeteorCreate(void* ptr_event_list)
{
	std::vector<const Event*>* event_list = (std::vector<const Event*>*)ptr_event_list;
	return new Meteor(event_list->size(), event_list->data());
}

void MeteorDestroy(void* ptr)
{
	delete (Meteor*)ptr;
}

void MeteorSaveToFile(void* ptr, const char* filename)
{
	Meteor* meteor = (Meteor*)ptr;
	meteor->SaveToFile(filename);
}

void MeteorLoadFromFile(void* ptr, const char* filename)
{
	Meteor* meteor = (Meteor*)ptr;
	meteor->LoadFromFile(filename);
}

void MeteorPlay(void* ptr_meteor, void* ptr_track)
{
	Meteor* meteor = (Meteor*)ptr_meteor;
	TrackBuffer* track = (TrackBuffer*)ptr_track;

	meteor->Play(track);
}

void* Base64Create(void* ptr_meteor)
{
	Meteor* meteor = (Meteor*)ptr_meteor;
	std::string* base64 = new std::string;
	meteor->ToBase64(*base64);
	return base64;
}

void Base64Destroy(void* ptr)
{
	delete (std::string*)ptr;
}

const char* Base64Get(void* ptr)
{
	return ((std::string*)ptr)->c_str();
}