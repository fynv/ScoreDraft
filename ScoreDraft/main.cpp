#include <stdio.h>
#include <string.h>

#include "parser/Parser.h"
#include "Note.h"
#include "TrackBuffer.h"
#include "instruments/PureSin.h"
#include "instruments/Square.h"
#include "instruments/Sawtooth.h"
#include "instruments/Triangle.h"
#include "instruments/NaivePiano.h"
#include "instruments/BottleBlow.h"

#include "Beat.h"

#ifdef _WIN32
#include "WinPCMPlayer.h"
#endif
#include "WinWavWriter.h"
#include "MIDIWriter.h"

#include "parser/CustomParser.h"

void Composite(const Document& doc, TrackBuffer& buffer)
{
	Sawtooth inst;

	size_t numTracks = doc.m_tracks.size();
	TrackBuffer_deferred *tracks = new TrackBuffer_deferred[numTracks];

	for (size_t i = 0; i < doc.m_tracks.size(); i++)
	{
		inst.PlayNotes(*tracks[i], *doc.m_tracks[i].m_note_seq, doc.m_tempo, doc.m_RefFreq);
		tracks[i]->SetVolume(doc.m_tracks[i].m_vol);
	}

	TrackBuffer::CombineTracks(buffer, (unsigned)numTracks, tracks);

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

#ifdef _WIN32

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
#endif

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



void test();
void kimiwo_nosete_just();
void AirBird_just();
void AirBird_eq();

void FlyMeToTheMoon_just();
void FlyMeToTheMoon_eq();

void BBD();

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		//test();
		AirBird_eq();
		AirBird_just();
		//FlyMeToTheMoon_just();
		//FlyMeToTheMoon_eq();
		//BBD();

		return 0;
	}

	string option = argv[1];

#ifdef _WIN32
	if (option == "-play")
	{
		if (argc>2)
			PlayFile(argv[2]);
	}
	else 
#endif
	if (option == "-wav")
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
		4.0f / 3.0f,
		//21.0f/16.0f,
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
	//doc.m_tempo = 150;
	doc.m_tempo = 80;

	CustomParser parser;
	parser.Customize(7, names, freqs);

	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	Track track2;
	NoteSequence& seq2 = *track2.m_note_seq;

	parser.ParseSeq("do6.24 re6.24 mi6.144 fa6.12 mi6.12 re6.12 do6.12 re6.24 so6.120 fa6.24 mi6.24 do6.144", seq1);

	//parser.ParseSeq("so5.24 la5.24 ti5.24 do6.24 re6.24 mi6.24 fa6.72 mi6.24 re6.48 do6.48 so5.48 mi5.48 do5.48", seq1);
	//parser.ParseSeq("so4.144 BK144 ti4.144 BK144 re5.144 so4.144 BK144 re5.144 BK144 fa5.144 so4.48 BK48 do5.48 BK48 mi5.48 BL96 do4.48 BK48 so4.48",seq2);

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	WriteToWav(tb, "test_just.wav");
}

void BBD()
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
		10.0f / 9.0f,
		5.0f / 4.0f,
		4.0f / 3.0f,
		3.0f / 2.0f,
		5.0f / 3.0f,
		50.0f / 27.0f
	};

	Document doc;
	doc.m_RefFreq = 264.0f *0.8f;
	doc.m_tempo = 85;

	CustomParser parser;
	parser.Customize(7, names, freqs);

	Track track1;
	NoteSequence& seq1 = *track1.m_note_seq;

	Track track2;
	NoteSequence& seq2 = *track2.m_note_seq;

	parser.ParseSeq("BL24 la5.24 la5.24 ti5.24", seq1);
	parser.ParseSeq("la4.48 BK48 do5.48 BK48 mi5.48 la4.48 BK48 do5.48 BK48 mi5.48", seq2);

	parser.ParseSeq("do6.24 so6.24 mi6.36 mi6.12", seq1);
	parser.ParseSeq("do5.48 BK48 mi5.48 BK48 so5.48 do5.48 BK48 mi5.48 BK48 so5.48", seq2);

	parser.ModifyFreq("so", 40.0f / 27.0f);
	parser.ParseSeq("re6.24 do6.24 re6.24 do6.12 re6.36", seq1);
	parser.ParseSeq("so4.48 BK48 ti4.48 BK48 re5.48 so4.48 BK48 ti4.48 BK48 re5.48", seq2);

	parser.ParseSeq("mi6.24 re6.24 do6.12 la5.12", seq1);
	parser.ParseSeq("re5.48 BK48 fa5.48 BK48 la5.48 re5.48 BK48 fa5.48 BK48 la5.48", seq2);

	doc.m_tracks.push_back(track1);
	doc.m_tracks.push_back(track2);

	TrackBuffer tb;
	Composite(doc, tb);

	WriteToWav(tb, "BBD.wav");

}
