#pragma once

#include <string>
#include <vector>

class Event
{
public:
	unsigned type;
	unsigned instrument_id;
	float offset;
	Event(unsigned type, unsigned instrument_id) : type(type), instrument_id(instrument_id)
	{

	}

	virtual ~Event()
	{

	}
};

class EventInst : public Event
{
public:
	double freq;
	float fduration;
	EventInst(unsigned instrument_id, double freq, float fduration)
		: Event(0, instrument_id)
		, freq(freq)
		, fduration(fduration)
	{

	}
};

class EventPerc : public Event
{
public:
	float fduration;
	EventPerc(unsigned instrument_id, float fduration)
		: Event(1, instrument_id)
		, fduration(fduration)
	{

	}
};

struct CtrlPnt
{
	double freq;
	double fduration;
};

struct Syllable
{
	std::string lyric;
	std::vector<CtrlPnt> ctrlPnts;
};

class EventSing : public Event
{
public:
	std::vector<Syllable> syllableList;
	EventSing(unsigned instrument_id, size_t num_syllables, const Syllable** syllables)
		: Event(2, instrument_id), syllableList(num_syllables)
	{
		for (size_t i = 0; i < num_syllables; i++)
		{
			syllableList[i] = *syllables[i];
		}
	}
};


struct VisNote
{
	unsigned instrumentId;
	int pitch;
	float start;
	float end;
};

struct VisBeat
{
	unsigned percId;
	float start;
	float end;
};

typedef std::vector<std::pair<int, float>> TempoMap;

struct VisSinging
{
	unsigned singerId;
	std::string lyric;
	std::vector<float> pitch;
	float start;
	float end;
};

class TrackBuffer;

#include "SubListLookUp.h"

class Meteor
{
	std::vector<VisNote> m_notes;
	std::vector<VisBeat> m_beats;
	std::vector<VisSinging> m_singings;

	SubLists m_notes_sublists;
	SubLists m_beats_sublists;
	SubLists m_singing_sublists;

	bool m_needUpdateSublists = true;
	void _updateSublists();

public:
	Meteor(){}
	~Meteor(){}

	Meteor(size_t num_events, const Event** events);

	void InstEvent(EventInst& e);
	void PercEvent(EventPerc& e);
	void SingEvent(EventSing& e);
	void SaveToFile(const char* filename);
	void LoadFromFile(const char* filename);

	const std::vector<VisNote>& GetNotes() const { return m_notes; }
	const std::vector<VisBeat>& GetBeats() const { return m_beats; }
	const std::vector<VisSinging>& GetSingings() const { return m_singings; }
	const SubLists& GetNotesSublists() const { return m_notes_sublists; }
	const SubLists& GetBeatsSublists() const { return m_beats_sublists; }
	const SubLists& GetSingingSublists() const { return m_singing_sublists; }

	void Play(TrackBuffer* buffer);
};

