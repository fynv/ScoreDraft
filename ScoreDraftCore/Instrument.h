#ifndef _scoredraft_Instrument_h
#define _scoredraft_Instrument_h

class  NoteBuffer
{
public:
	NoteBuffer();
	~NoteBuffer();
	
	unsigned m_sampleNum;
	float* m_data;
	void Allocate();	
};

class TrackBuffer;
class Note;
class NoteSequence;
class Instrument
{
public:
	Instrument();
	~Instrument();

	void PlayNote(TrackBuffer& buffer, const Note& aNote, unsigned tempo=80,float RefFreq=261.626f);
	void PlayNotes(TrackBuffer& buffer, const NoteSequence& seq, unsigned tempo=80,float RefFreq=261.626f);

	virtual bool Tune(const char* cmd);
	
protected:
	void Silence(unsigned numOfSamples, NoteBuffer* noteBuf);
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf);

	float m_noteVolume;

};

#endif
