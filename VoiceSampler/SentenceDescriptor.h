#ifndef __SentenceDescriptor_h
#define __SentenceDescriptor_h

#include <vector>

struct Wav
{
	float *buf;
	unsigned len;
};

struct FrqDataPoint
{
	double freq;
	double dyn;
};

struct FrqData
{
	int interval;
	double key;
	std::vector<FrqDataPoint> data;
};
struct Source
{
	Wav wav;
	FrqData frq;
};

struct SourceMapCtrlPnt
{
	float srcPos;
	float dstPos;
	int isVowel; // 0: notVowel, 1: preVowel, 2: isVowel
};

struct Piece
{
	Source src;
	std::vector<SourceMapCtrlPnt> srcMap;
};

struct GeneralCtrlPnt
{
	float value;
	float dstPos;
};

struct SentenceDescriptor
{
	std::vector<Piece> pieces;
	std::vector<GeneralCtrlPnt> piece_map;
	std::vector<GeneralCtrlPnt> freq_map;
	std::vector<GeneralCtrlPnt> volume_map;
};

#endif

