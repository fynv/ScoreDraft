#include <stdio.h>

#include "parser/Parser.h"
#include "Note.h"
#include "TrackBuffer.h"
#include "instruments/PureSin.h"
#include "instruments/Sawtooth.h"
#include "instruments/NaivePiano.h"

#include "WinPCMPlayer.h"
#include "WinWavWriter.h"
#include "MIDIWriter.h"

#include "parser/CustomParser.h"

void Composite(const Document& doc, TrackBuffer& buffer)
{
	NaivePiano inst;

	size_t numTracks = doc.m_tracks.size();
	TrackBuffer_deferred *tracks = new TrackBuffer_deferred[numTracks];

	for (size_t i = 0; i < doc.m_tracks.size(); i++)
	{
		inst.PlayNotes(*tracks[i], *doc.m_tracks[i].m_note_seq, doc.m_tempo, doc.m_RefFreq);
		tracks[i]->SetVolume(doc.m_tracks[i].m_vol);
	}

	TrackBuffer::CombineTracks(buffer, numTracks, tracks);

	float maxV = buffer.MaxValue();
	buffer.SetVolume(1.0f / maxV);

	delete[] tracks;
}

void ParseToDoc(const char* filename, Document& doc)
{
	Parser par;
	par.m_doc = &doc;

	FILE *fp = fopen(filename, "r");
	char lineBuffer[2048];
	string stringCmdLine;

	unsigned lineCount = 0;
	while (fgets(lineBuffer, 2048, fp))
	{
		lineCount++;
		if (lineBuffer[strlen(lineBuffer) - 1] == 10) lineBuffer[strlen(lineBuffer) - 1] = 0;
		bool finished = true;
		char* pChar = lineBuffer + strlen(lineBuffer) - 1;
		int pointCount = 0;
		while (pChar >= lineBuffer && *pChar == ' ' || *pChar == '\t' || *pChar == '.')
		{
			if (*pChar == '.')
			{
				pointCount++;
				if (pointCount == 3)
				{
					finished = false;
					break;
				}
			}
			else if (pointCount>0) break;
			pChar--;
		}
		if (!finished) *pChar = 0;
		stringCmdLine += lineBuffer;
		if (!finished) stringCmdLine += " ";

		if (finished)
		{
			char errMsg[1000];
			bool result = par.ParseLine(stringCmdLine.data(), errMsg);
			if (!result)
			{
				printf("Error Line %d: %s\n", lineCount, errMsg);
				break;
			}
			stringCmdLine = "";
		}
	}

	fclose(fp);
}

void PlayFile(const char* filename)
{
	Document doc;
	ParseToDoc(filename, doc);

	TrackBuffer tb;
	Composite(doc, tb);

	if (tb.NumberOfSamples() > 0)
	{

		WinPCMPlayer player;
		player.PlayTrack(tb);

		system("pause");
	}

}

void ToWav(const char* InFilename, const char* OutFileName)
{
	Document doc;
	ParseToDoc(InFilename, doc);

	TrackBuffer tb;
	Composite(doc, tb);

	WriteToWav(tb, OutFileName);
}

void ToMIDI(const char* InFilename, const char* OutFileName)
{
	Document doc;
	ParseToDoc(InFilename, doc);

	WriteToMidi(doc, OutFileName);
}



void kimiwo_nosete();
void test();

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		//kimiwo_nosete();
		test();
		return 0;
	}

	string option = argv[1];

	if (option == "-play")
	{
		if (argc>2)
			PlayFile(argv[2]);
	}
	else if (option == "-wav")
	{
		if (argc > 2)
		{
			string inFileName = argv[2];
			string outFileName;

			if (argc > 3) outFileName = argv[3];
			else
			{
				size_t pos=inFileName.rfind('.');
				outFileName = inFileName.substr(0, pos)+".wav";
			}

			ToWav(inFileName.data(), outFileName.data());
		}		
	}
	else if (option == "-midi")
	{
		if (argc > 2)
		{
			string inFileName = argv[2];
			string outFileName;

			if (argc > 3) outFileName = argv[3];
			else
			{
				size_t pos = inFileName.rfind('.');
				outFileName = inFileName.substr(0, pos) + ".mid";
			}

			ToMIDI(inFileName.data(), outFileName.data());
		}
	}

	return 0;
}

void kimiwo_nosete()
{
	static const char* names[] =
	{
		"do",
		"re",
		"+re",
		"mi",
		"fa",
		"+fa",
		"so",
		"+so",
		"la",
		"ti"
	};

	static float freqs[] =
	{
		1.0f,
		10.0f / 9.0f,
		75.0f / 64.0f,
		5.0f / 4.0f,
		4.0f / 3.0f,
		45.0f / 32.0f,
		3.0f / 2.0f,
		25.0f / 16.0f,
		5.0f / 3.0f,
		15.0f / 8.0f
	};

	Document doc;
	doc.m_RefFreq = 2475.0f / 8.0f; // bE
	doc.m_tempo = 100;

	CustomParser parser;
	parser.Customize(10, names, freqs);

	NoteSequence Am;
	parser.ParseSeq("la3.192 BK168 mi4.24 la4.24 do5.24 la4.24 do5.24 la4.24 mi4.24", Am);

	NoteSequence Em;
	parser.ParseSeq("mi3.192 BK168 ti3.24 mi4.24 so4.24 mi4.24 so4.24 mi4.24 ti3.24", Em);

	NoteSequence Dm;
	parser.ParseSeq("re3.192 BK168 la3.24 re4.24 fa4.24 re4.24 fa4.24 re4.24 la3.24", Dm);

	parser.ModifyFreq("re", 9.0f / 8.0f);
	
	NoteSequence C;
	parser.ParseSeq("do3.192 BK168 so3.24 do4.24 mi4.24 do4.24 mi4.24 do4.24 so3.24", C);

	NoteSequence G;
	parser.ParseSeq("so3.192 BK168 re4.24 so4.24 ti4.24 so4.24 ti4.24 so4.24 re4.24", G);

	NoteSequence F;
	parser.ParseSeq("fa3.192 BK168 do4.24 fa4.24 la4.24 fa4.24 la4.24 fa4.24 do4.24", F);

	NoteSequence E;
	parser.ParseSeq("mi3.192 BK168 ti3.24 mi4.24 +so4.24 mi4.24 +so4.24 mi4.24 ti3.24", E);

	NoteSequence B;
	parser.ParseSeq("ti3.192 BK168 +fa4.24 ti4.24 +re5.24 ti4.24 +re5.24 ti4.24 +fa4.24", B);
	
	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	NoteSequence main;

	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("la5.24 ti5.24 do6.72 ti5.24 do6.48 mi6.48 ti5.96 BL48", main);
	parser.ParseSeq("mi5.48        la5.72 so5.24 la5.48 do6.48 so5.96 BL48", main);
	parser.ParseSeq("mi5.48        fa5.72 mi5.24 fa5.48 do6.48 mi5.96 BL48", main);
	parser.ParseSeq("do6.48        ti5.72 +fa5.24 +fa5.48 ti5.48 ti5.96 BL48", main);
	parser.ParseSeq("la5.24 ti5.24 do6.72 ti5.24 do6.48 mi6.48 ti5.96 BL48", main);
	parser.ParseSeq("so5.48        la5.72 so5.24 la5.48 do6.48 so5.96 BL48", main);
	parser.ParseSeq("mi5.48        fa5.72 do6.24 ti5.48 do6.48 re6.48 mi6.24 do6.72", main);
	parser.ParseSeq("BL48          do6.24 ti5.24 la5.48 ti5.48 +so5.48 la5.144", main);

	seq1 = main;

	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ParseSeq("do6.24 re6.24 mi6.72 re6.24 mi6.48 so6.48 re6.96  BL48", seq1);
	parser.ParseSeq("so5.48        do6.72 ti5.24 do6.48 mi6.48 mi6.144 BL48", seq1);
	parser.ParseSeq("la5.24 ti5.24 do6.48 ti5.24 do6.24 re6.24 re6.24  do6.72 so5.24 so5.96", seq1);
	parser.ModifyFreq("re", 10.0f / 9.0f);
	parser.ParseSeq("fa6.48 mi6.48 re6.48 do6.48                       mi6.144", seq1);

	parser.ParseSeq("mi6.48        la6.96 so6.96   mi6.24 re6.24 do6.96", seq1);
	parser.ModifyFreq("re", 9.0f / 8.0f);
	parser.ParseSeq("BL48          re6.72 do6.24   re6.48 so6.48 mi6.144", seq1);
	parser.ParseSeq("mi6.48        la6.96 so6.96   mi6.24 re6.24 do6.96", seq1);
	parser.ParseSeq("BL48          re6.72 do6.24   re6.48 ti5.48 la5.144", seq1);

	seq1 += main;

	Track track2;
	track2.m_vol = 0.6f;
	NoteSequence& seq2 = *track2.m_note_seq;
	parser.ParseSeq("BL48",seq2);

	NoteSequence accompany_main;
	
	accompany_main += Am + Em;
	accompany_main += F + C;
	accompany_main += Dm + C;
	accompany_main += B + E;
	accompany_main += Am + G;
	accompany_main += F + C;
	accompany_main += Dm + Am;
	accompany_main += E + Am;

	seq2 += accompany_main;

	seq2 += C + G;
	seq2 += Am + Em;
	seq2 += F + C;
	seq2 += Dm + E;

	seq2 += Am + F;
	seq2 += G + C;
	seq2 += Am + F;
	seq2 += G + Am;

	seq2 += accompany_main;

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	/*WinPCMPlayer player;
	player.PlayTrack(tb);

	system("pause");*/

	WriteToWav(tb, "kimiwo nosete2_just.wav");
}

void test()
{
	static const char* names[] =
	{
		"do",
		"re",
		"mi",
		"fa",
		"so",
		"la",
		"ti"
	};

	static float freqs[] =
	{
		1.0f,
		9.0f / 8.0f,
		5.0f / 4.0f,
		//4.0f / 3.0f,
		21.0f/16.0f,
		3.0f / 2.0f,
		5.0f / 3.0f,
		15.0f / 8.0f
	};

	/*static float freqs[] =
	{
		1.0f,
		powf(2.0f, 2.0f/12.0f),
		powf(2.0f, 4.0f / 12.0f),
		powf(2.0f, 5.0f / 12.0f),
		powf(2.0f, 7.0f / 12.0f),
		powf(2.0f, 9.0f / 12.0f),
		powf(2.0f, 11.0f / 12.0f),
	};*/

	Document doc;
	doc.m_RefFreq = 264.0f;
	doc.m_tempo = 150;

	CustomParser parser;
	parser.Customize(7, names, freqs);

	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	Track track2;
	NoteSequence& seq2 = *track2.m_note_seq;

	parser.ParseSeq("so5.24 la5.24 ti5.24 do6.24 re6.24 mi6.24 fa6.72 mi6.24 re6.48 do6.48 so5.48 mi5.48 do5.48", seq1);
	parser.ParseSeq("so4.144 BK144 ti4.144 BK144 re5.144 so4.144 BK144 re5.144 BK144 fa5.144 so4.48 BK48 do5.48 BK48 mi5.48 BL96 do4.48 BK48 so4.48",seq2);

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	/*WinPCMPlayer player;
	player.PlayTrack(tb);

	system("pause");*/

	WriteToWav(tb, "test_harmonic.wav");
}
	


