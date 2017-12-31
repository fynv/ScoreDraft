#include "parser/Parser.h"
#include "parser/NoteSeqTable.h"
#include "Note.h"
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <stdio.h>

Parser::Parser()
{
	m_nst = new NoteSeqTable;
	m_doc = NULL;
}

Parser::~Parser()
{
	delete m_nst;
}


static char* names[] =
{
	"do",
	"re",
	"mi",
	"fa",
	"so",
	"la",
	"ti",
	"BL",
	"BK"
};

static float shifts[] =
{
	0.0f,
	2.0f,
	4.0f,
	5.0f,
	7.0f,
	9.0f,
	11.0f
};

/*static float shifts[] =
{
	0.0f,
	//1.824f,
	2.039f,
	3.863f,
	4.980f,
	7.020f,
	8.844f,
	10.883f
};*/

bool Parser::ParseNote(const char* strNote, Note& note)
{
	if (strlen(strNote)<3) return false;

	char name[3];
	name[0] = strNote[0]; name[1] = strNote[1]; name[2] = 0;
	string stringName = name;

	int i;
	for (i = 0; i<9; i++)
	{
		string stringName2 = names[i];
		if (stringName == stringName2) break;
	}
	if (i >= 9) return false;

	if (i == 7)
	{
		const char* pDur = strNote + 2;
		note.m_freq_rel = -1.0f;
		note.m_duration = atoi(pDur);
		return true;
	}
	else if (i == 8)
	{
		const char* pDur = strNote + 2;
		note.m_freq_rel = -1.0f;
		note.m_duration = -atoi(pDur);
		return true;
	}

	float shift = shifts[i];

	const char* pOct = strNote + 2;

	if (pOct[0] == '+')
	{
		if (i == 2 || i == 6) return false;
		shift += 1.0f;
		pOct++;
	}
	else if (pOct[0] == '-')
	{
		if (i == 0 || i == 3) return false;
		shift -= 1.0f;
		pOct++;
	}

	const char* pEdOct = strchr(pOct, '.');
	if (!pEdOct) return false;

	char strOct[10];

	for (i = 0; pOct != pEdOct; i++, pOct++)
		strOct[i] = *pOct;
	strOct[i] = 0;

	unsigned Oct = atoi(strOct);
	note.m_freq_rel = powf(2.0f, ((float)(12 * Oct) + shift-60.0f)/12.0f);

	const char* pDur = pEdOct + 1;
	note.m_duration = atoi(pDur);

	return true;
}

bool Parser::ParseSeq(const char* strSeq, NoteSequence& seq, char* errMsg)
{
	std::vector<string> subStrings;

	char subStr[100];

	while (sscanf(strSeq, "%s", subStr) != EOF)
	{
		string subString = subStr;
		subStrings.push_back(subString);
		strSeq = strstr(strSeq, subStr) + strlen(subStr);
	}

	bool* IsNote = new bool[subStrings.size()];
	unsigned NumOfNotes = 0;
	unsigned i;
	for (i = 0; i<subStrings.size(); i++)
	{
		const NoteSequence* subSeq = m_nst->FindNoteSequence(subStrings[i]);
		IsNote[i] = (subSeq == 0);
		if (subSeq) NumOfNotes += (unsigned)subSeq->size();
		else NumOfNotes++;
	}
	size_t originalSize = seq.size();
	seq.resize(NumOfNotes + originalSize);

	unsigned j = 0;

	for (i = 0; i<subStrings.size(); i++)
	{
		if (IsNote[i])
		{
			bool result = ParseNote(subStrings[i].data(), seq[j + originalSize]);
			if (!result)
			{
				if (errMsg) sprintf(errMsg, "Note parse failure: %s", subStrings[i].data());
				seq.clear();
				return false;
			}
			j++;
		}
		else
		{
			const NoteSequence& subSeq = *m_nst->FindNoteSequence(subStrings[i]);
			unsigned k;
			for (k = 0; k<subSeq.size(); k++, j++)
				seq[j + originalSize] = subSeq[k];
		}
	}

	delete[] IsNote;
	return true;
}

bool Parser::ParseSet(const char* strSet, char* errMsg)
{
	const char* charEq = strchr(strSet, '=');
	if (!charEq) return false;

	char _strName[100];
	memcpy(_strName, strSet, charEq - strSet);
	_strName[charEq - strSet] = 0;

	char strName[100];
	sscanf(_strName, "%s", strName);

	string stringName = strName;

	const char* strSeq = charEq + 1;

	NoteSequence_deferred nseq;
	bool result = ParseSeq(strSeq, *nseq, errMsg);
	if (!result) return false;

	Note testNote;
	if (ParseNote(strName, testNote))
	{
		if (errMsg) sprintf(errMsg, "Sequence set failure, name of sequence \" %s \"can be parsed as a note.", strName);
		return false;
	}

	if (!m_nst->AddNoteSequence(stringName, nseq))
	{
		if (errMsg) sprintf(errMsg, "Sequence set failure, name of sequence \" %s \" already used.", strName);
		return false;
	}

	return true;
}

bool Parser::ParseTrack(const char* strTrack, char* errMsg)
{
	const char* strSeqLabel = strstr(strTrack, "seq");
	if (!strSeqLabel)
	{
		if (errMsg) sprintf(errMsg, "Track parse failure.");
		return false;
	}
	const char* strEq = strchr(strSeqLabel + 3, '=');
	if (!strEq)
	{
		if (errMsg) sprintf(errMsg, "Track parse failure.");
		return false;
	}
	const char* strQuot1 = strchr(strEq + 1, '\"');
	if (!strQuot1)
	{
		if (errMsg) sprintf(errMsg, "Track parse failure.");
		return false;
	}
	const char* strQuot2 = strchr(strQuot1 + 1, '\"');
	if (!strQuot2)
	{
		if (errMsg) sprintf(errMsg, "Track parse failure.");
		return false;
	}

	char *strSeq = new char[strQuot2 - strQuot1];
	strSeq[strQuot2 - strQuot1 - 1] = 0;
	memcpy(strSeq, strQuot1 + 1, strQuot2 - strQuot1 - 1);

	NoteSequence_deferred nseq;
	bool result = ParseSeq(strSeq, *nseq, errMsg);
	delete[] strSeq;

	if (!result) return false;

	char instName[100];
	instName[0] = 0;
	float vol = 1.0f;

	const char* strInstLabel = strstr(strTrack, "inst");
	if (strInstLabel)
	{
		strEq = strchr(strInstLabel + 4, '=');
		if (!strEq)
		{
			if (errMsg) sprintf(errMsg, "Track parse failure.");
			return false;
		}
		sscanf(strEq + 1, "%s", instName);
	}

	const char* strVolLabel = strstr(strTrack, "vol");
	if (strVolLabel)
	{
		strEq = strchr(strVolLabel + 3, '=');
		if (!strEq)
		{
			if (errMsg) sprintf(errMsg, "Track parse failure.");
			return false;
		}
		sscanf(strEq + 1, "%f", &vol);
	}

	if (m_doc)
	{
		Track track;
		track.m_note_seq = nseq;
		track.m_instrument_name = instName;
		track.m_vol = vol;

		m_doc->m_tracks.push_back(track);
	}
	return true;
}

bool Parser::ParseLine(const char* strLine, char* errMsg)
{
	string stringLine = strLine;
	int notation = (int) stringLine.find_first_of('#');
	if (notation>=0)
	{
		stringLine = string(strLine, notation);
		strLine = stringLine.data();
	}

	char typeName[100];
	if (sscanf(strLine, "%s", typeName) == EOF) return true;

	string stringTypeName = typeName;

	if (stringTypeName == "set")
	{
		const char* strSet = strstr(strLine, "set") + 3;
		return ParseSet(strSet, errMsg);
	}

	if (stringTypeName == "track")
	{
		const char* strTrack = strstr(strLine, "track") + 5;
		return ParseTrack(strTrack, errMsg);
	}

	if (stringTypeName == "tempo")
	{
		const char* strTempo = strstr(strLine, "tempo") + 5;
		unsigned tempo;
		sscanf(strTempo, "%d", &tempo);
		if (tempo>0)
		{
			if (m_doc)
			{
				m_doc->m_tempo = tempo;
			}
			return true;
		}
		else
		{
			if (errMsg) sprintf(errMsg, "Bad tempo.");
			return false;
		}
	}

	if (stringTypeName == "reffreq")
	{
		const char* strReffreq = strstr(strLine, "reffreq") + 7;
		float reffreq;
		sscanf(strReffreq, "%f", &reffreq);
		if (reffreq>0.0f)
		{
			if (m_doc)
			{
				m_doc->m_RefFreq = reffreq;
			}
			return true;
		}
		else
		{
			if (errMsg) sprintf(errMsg, "Bad reference frequency.");
			return false;
		}
	}

	if (errMsg) sprintf(errMsg, "Bad command!");

	return false;
}
