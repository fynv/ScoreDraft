#ifndef _scoredraft_Instrument_h
#define _scoredraft_Instrument_h

#include <vector>
#include <string>

class NoteBuffer;
class TrackBuffer;
class Note;
class NoteSequence;

typedef std::vector<std::pair<int, float>> TempoMap;

class Instrument
{
public:
	Instrument();
	virtual ~Instrument();

	void PlayNote(TrackBuffer& buffer, const Note& aNote, unsigned tempo=80,float RefFreq=261.626f);
	void PlayNote(TrackBuffer& buffer, const Note& aNote, const TempoMap& tempoMap, int tempoMapOffset = 0, float RefFreq = 261.626f);

	virtual bool Tune(const char* cmd);

	void SetNoteVolume(float volume) {	m_noteVolume = volume;	}
	void SetNotePan(float pan) { m_notePan = pan;  }

	virtual bool IsGMDrum() { return false; }
	
protected:
	void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	float m_noteVolume;
	float m_notePan;

};

#endif
