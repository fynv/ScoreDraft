#include "Meteor.h"
#include "blob.hpp"
#include "base64.hpp"

void Meteor::_updateSublists()
{
	m_notes_sublists.SetData(m_notes, 3.0f);
	m_beats_sublists.SetData(m_beats, 3.0f);
	m_singing_sublists.SetData(m_singings, 3.0f);
	m_needUpdateSublists = false;
}

void Meteor::InstEvent(EventInst& e)
{
	VisNote note;
	note.instrumentId = e.instrument_id;
	double freq = e.freq;		
	note.pitch = (int)floor(log(freq / 261.626)*12.0 / log(2.0) + 0.5);
	note.start = e.offset / 1000.0f;
	note.end = note.start + e.fduration / 1000.0f;
	m_notes.push_back(note);
	m_needUpdateSublists = true;
}

void Meteor::PercEvent(EventPerc& e)
{
	VisBeat beat;
	beat.percId = e.instrument_id;
	beat.start = e.offset / 1000.0f;
	beat.end = beat.start + e.fduration / 1000.0;
	m_beats.push_back(beat);
	m_needUpdateSublists = true;
}

void Meteor::SingEvent(EventSing& e)
{
	unsigned id = e.instrument_id;	
	unsigned count = (unsigned)e.syllableList.size();
	double pos = e.offset / 1000.0f;
	for (unsigned i = 0; i < count; i++)
	{
		Syllable& syllable = e.syllableList[i];

		VisSinging sing;
		sing.singerId = id;
		sing.lyric = syllable.lyric;
		sing.start = (float)pos;
		
		std::vector<CtrlPnt>& ctrlPnts = syllable.ctrlPnts;
		unsigned num_ctrlPnts = (unsigned)ctrlPnts.size();

		double totalLen = 0.0;
		std::vector<double> freqs;
		std::vector<double> durations;
		freqs.resize(num_ctrlPnts);
		durations.resize(num_ctrlPnts);
		for (unsigned j = 0; j < num_ctrlPnts; j++)
		{
			CtrlPnt& ctrlPnt = ctrlPnts[j];
			double freq = ctrlPnt.freq;				
			double duration = ctrlPnt.fduration / 1000.0;
			freqs[j] = freq;
			durations[j] = duration;
			totalLen += duration;
		}

		sing.end = sing.start + totalLen;
		double pitchPerSec = 50.0;
		double secPerPitchSample = 1.0 / pitchPerSec;

		unsigned pitchLen = (unsigned)ceil(totalLen*pitchPerSec);
		sing.pitch.resize(pitchLen);
		unsigned pitchPos = 0;
		double fPitchPos = 0.0;
		double nextPitchPos = 0.0;

		for (unsigned j = 0; j < num_ctrlPnts; j++)
		{
			double duration = durations[j];
			double freq1 = freqs[j];
			double freq2 = j < num_ctrlPnts - 1 ? freqs[j + 1] : freq1;
			double thisPitchPos = nextPitchPos;
			nextPitchPos += duration;

			for (; fPitchPos < nextPitchPos && pitchPos < pitchLen; fPitchPos += secPerPitchSample, pitchPos++)
			{
				double k = (fPitchPos - thisPitchPos) / duration;
				sing.pitch[pitchPos] = (float)(freq1*(1.0f - k) + freq2 * k);
			}
		}

		std::vector<float> temp;
		temp.resize(pitchLen);
		temp[0] = sing.pitch[0];
		temp[pitchLen - 1] = sing.pitch[pitchLen - 1];

		for (unsigned i = 1; i < pitchLen - 1; i++)
		{
			temp[i] = 0.25f*(sing.pitch[i - 1] + sing.pitch[i + 1]) + 0.5f*sing.pitch[i];
		}

		for (unsigned i = 0; i < pitchLen; i++)
		{
			sing.pitch[i] = logf(temp[i] / 261.626f)*12.0f / logf(2.0f);
		}

		m_singings.push_back(sing);
		pos += totalLen;
	}
	m_needUpdateSublists = true;
}

void Meteor::SaveToFile(const char* filename)
{
	if (m_needUpdateSublists) _updateSublists();

	FILE *fp = fopen(filename, "wb");
	unsigned countNotes = (unsigned)m_notes.size();
	fwrite(&countNotes, sizeof(unsigned), 1, fp);
	fwrite(&m_notes[0], sizeof(VisNote), countNotes, fp);
	m_notes_sublists.SaveToFile(fp);
	unsigned countBeats = (unsigned)m_beats.size();
	fwrite(&countBeats, sizeof(unsigned), 1, fp);
	fwrite(&m_beats[0], sizeof(VisBeat), countBeats, fp);
	m_beats_sublists.SaveToFile(fp);
	unsigned countSinging = (unsigned)m_singings.size();
	fwrite(&countSinging, sizeof(unsigned), 1, fp);
	for (unsigned i = 0; i < countSinging; i++)
	{
		const VisSinging& singing = m_singings[i];
		fwrite(&singing.singerId, sizeof(unsigned), 1, fp);
		unsigned char len = (unsigned char)singing.lyric.length();
		fwrite(&len, 1, 1, fp);
		if (len > 0)
			fwrite(singing.lyric.data(), 1, len, fp);
		unsigned count = (unsigned)singing.pitch.size();
		fwrite(&count, sizeof(unsigned), 1, fp);
		fwrite(singing.pitch.data(), sizeof(float), count, fp);
		fwrite(&singing.start, sizeof(float), 2, fp);
	}
	m_singing_sublists.SaveToFile(fp);
	fclose(fp);
}

void Meteor::LoadFromFile(const char* filename)
{
	FILE *fp = fopen(filename, "rb");
	unsigned countNotes;
	fread(&countNotes, sizeof(unsigned), 1, fp);
	m_notes.clear();
	m_notes.resize(countNotes);
	fread(&m_notes[0], sizeof(VisNote), countNotes, fp);
	m_notes_sublists.LoadFromFile(fp);
	unsigned countBeats;
	fread(&countBeats, sizeof(unsigned), 1, fp);
	m_beats.clear();
	m_beats.resize(countBeats);
	fread(&m_beats[0], sizeof(VisBeat), countBeats, fp);
	m_beats_sublists.LoadFromFile(fp);
	unsigned countSinging;
	fread(&countSinging, sizeof(unsigned), 1, fp);
	m_singings.clear();
	m_singings.resize(countSinging);
	for (unsigned i = 0; i < countSinging; i++)
	{
		VisSinging& singing = m_singings[i];
		fread(&singing.singerId, sizeof(unsigned), 1, fp);
		unsigned char len;
		fread(&len, 1, 1, fp);
		if (len > 0)
		{
			char _str[256];
			fread(_str, 1, len, fp);
			_str[len] = 0;
			singing.lyric = _str;
		}
		else
			singing.lyric = "";
		unsigned count;
		fread(&count, sizeof(unsigned), 1, fp);
		singing.pitch.resize(count);
		fread(singing.pitch.data(), sizeof(float), count, fp);
		fread(&singing.start, sizeof(float), 2, fp);
	}
	m_singing_sublists.LoadFromFile(fp);
	fclose(fp);
	m_needUpdateSublists = true;
}


void Meteor::ToBlob(std::vector<uint8_t>& blob)
{
	if (m_needUpdateSublists) _updateSublists();

	blob.clear();
	unsigned countNotes = (unsigned)m_notes.size();
	blob_write(blob, &countNotes, sizeof(unsigned));
	blob_write(blob, &m_notes[0], sizeof(VisNote) * countNotes);
	m_notes_sublists.ToBlob(blob);
	unsigned countBeats = (unsigned)m_beats.size();
	blob_write(blob, &countBeats, sizeof(unsigned));
	blob_write(blob, &m_beats[0], sizeof(VisBeat)*countBeats);
	m_beats_sublists.ToBlob(blob);
	unsigned countSinging = (unsigned)m_singings.size();
	blob_write(blob, &countSinging, sizeof(unsigned));
	for (unsigned i = 0; i < countSinging; i++)
	{
		const VisSinging& singing = m_singings[i];
		blob_write(blob, &singing.singerId, sizeof(unsigned));
		unsigned char len = (unsigned char)singing.lyric.length();
		blob_write(blob, &len, 1);
		if (len > 0)
			blob_write(blob, singing.lyric.data(), len);
		unsigned count = (unsigned)singing.pitch.size();
		blob_write(blob, &count, sizeof(unsigned));
		blob_write(blob, singing.pitch.data(), sizeof(float) * count);
		blob_write(blob, &singing.start, sizeof(float)*2);
	}
	m_singing_sublists.ToBlob(blob);
}

void Meteor::ToBase64(std::string& base64)
{
	std::vector<uint8_t> blob;
	ToBlob(blob);
	base64_encode(blob, base64);
}

Meteor::Meteor(size_t num_events, const Event** events)
{
	for (unsigned i = 0; i < num_events; i++)
	{
		const Event* e = events[i];
		switch (e->type)
		{
		case 0:
			InstEvent(*(EventInst*)e);
			break;
		case 1:
			PercEvent(*(EventPerc*)e);
			break;
		case 2:
			SingEvent(*(EventSing*)e);
			break;
		}
	}

}

#include "MeteorPlayer.h"

void Meteor::Play(TrackBuffer* buffer)
{
	if (m_needUpdateSublists)
		_updateSublists();

	MeteorPlayer player(this, buffer);
	player.main_loop();
}