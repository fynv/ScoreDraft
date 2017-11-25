#ifndef _scoredraft_Document_h
#define _scoredraft_Document_h

#include <string>
#include <vector>
#include "Deferred.h"

using std::string;

class NoteSequence;
typedef Deferred<NoteSequence> NoteSequence_deferred;

class Track
{
public:
	Track():m_vol(1.0f){}
	NoteSequence_deferred m_note_seq;
	string m_instrument_name;
	float m_vol;
};

typedef std::vector<Track> TrackList;

class Document
{
public:
	Document()
	{
		m_tempo = 80;
		m_RefFreq = 261.626f;
	}
	unsigned m_tempo;
	float m_RefFreq;
	TrackList m_tracks;
};

#endif

