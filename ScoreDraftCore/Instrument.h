#ifndef _scoredraft_Instrument_h
#define _scoredraft_Instrument_h

class NoteBuffer;
class TrackBuffer;
class Note;
class NoteSequence;
class Instrument
{
public:
	Instrument();
	virtual ~Instrument();

	void PlayNote(TrackBuffer& buffer, const Note& aNote, unsigned tempo=80,float RefFreq=261.626f);

	virtual bool Tune(const char* cmd);
	
protected:
	void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	float m_noteVolume;
	float m_notePan;

};

#endif
