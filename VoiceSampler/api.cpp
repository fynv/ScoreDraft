#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	SCOREDRAFT_API unsigned HaveCUDA();
	SCOREDRAFT_API void* FrqDataPointCreate(double freq, double dyn);
	SCOREDRAFT_API void FrqDataPointDestroy(void* ptr);
	SCOREDRAFT_API void* FrqDataCreate();
	SCOREDRAFT_API void FrqDataDestroy(void* ptr);
	SCOREDRAFT_API void FrqDataSet(void* ptr, int interval, double key, void* ptr_data_points);
	SCOREDRAFT_API void FrqDataDetect(void* ptr, void* ptr_f32_buf, int interval);
	SCOREDRAFT_API void* SourceMapCtrlPntCreate(float srcPos, float dstPos, int isVowel);
	SCOREDRAFT_API void SourceMapCtrlPntDestroy(void* ptr);
	SCOREDRAFT_API void* PieceCreate(void* ptr_f32buf, void* ptr_frq_data, void* ptr_src_map);
	SCOREDRAFT_API void PieceDestroy(void* ptr);
	SCOREDRAFT_API void* GeneralCtrlPntCreate(float value, float dstPos);
	SCOREDRAFT_API void GeneralCtrlPntDestroy(void* ptr);
	SCOREDRAFT_API void* SentenceDescriptorCreate(void* ptr_pieces, void* ptr_piece_map, void* ptr_freq_map, void* ptr_volume_map);
	SCOREDRAFT_API void SentenceDescriptorDestroy(void* ptr);
	SCOREDRAFT_API void GenerateSentence(void* ptr_wavbuf, void* ptr_sentence);
	SCOREDRAFT_API void GenerateSentenceCUDA(void* ptr_wavbuf, void* ptr_sentence);
}

#include "utils.h"
#include <WavBuffer.h>
#include "SentenceDescriptor.h"
#include "SentenceGeneratorCPU.h"
#ifdef HAVE_CUDA
#include "SentenceGeneratorCUDA.h"
#include <cuda_runtime.h>
#endif
#include <cmath>

#include "VoiceUtil.h"
#include "FrequencyDetection.h"
using namespace VoiceUtil;

static bool s_have_cuda = false;
unsigned HaveCUDA()
{
#if HAVE_CUDA
	int count;
	cudaGetDeviceCount(&count);
	if (count > 0 && cudaGetLastError() == 0)
	{
		cudaFree(nullptr);
		if (cudaGetLastError() == 0) s_have_cuda = true;
	}
#endif
	return s_have_cuda?1:0;
}

void* FrqDataPointCreate(double freq, double dyn)
{
	FrqDataPoint *pnt = new FrqDataPoint;
	pnt->freq = freq;
	pnt->dyn = dyn;
	return pnt;
}

void FrqDataPointDestroy(void* ptr)
{
	delete (FrqDataPoint*)ptr;
}

void* FrqDataCreate()
{
	return new FrqData;
}

void FrqDataDestroy(void* ptr)
{
	delete (FrqData*)ptr;
}

void FrqDataSet(void* ptr, int interval, double key, void* ptr_data_points)
{
	FrqData* frq_data = (FrqData*)ptr;
	PtrArray* data_points = (PtrArray*)ptr_data_points;	
	frq_data->interval = interval;
	frq_data->key = key;
	frq_data->data.resize(data_points->size());
	for (size_t i = 0; i < data_points->size(); i++)
	{
		FrqDataPoint *pnt = (FrqDataPoint *)(*data_points)[i];
		frq_data->data[i] = *pnt;
	}	
}


void DetectFreqs(const Buffer& buf, std::vector<float>& frequencies, std::vector<float>& dynamics, unsigned step)
{
	unsigned halfWinLen = 1024;
	float* temp = new float[halfWinLen * 2];

	for (unsigned center = 0; center < buf.m_data.size(); center += step)
	{
		Window win;
		win.CreateFromBuffer(buf, (float)center, (float)halfWinLen);

		for (int i = -(int)halfWinLen; i < (int)halfWinLen; i++)
			temp[i + halfWinLen] = win.GetSample(i);

		float freq;
		float dyn;
		fetchFrequency(halfWinLen * 2, temp, buf.m_sampleRate, freq, dyn);

		frequencies.push_back(freq);
		dynamics.push_back(dyn);
	}

	delete[] temp;
}

void FrqDataDetect(void* ptr, void* ptr_f32_buf, int interval)
{
	FrqData* frq_data = (FrqData*)ptr;
	F32Buf* f32buf = (F32Buf*)ptr_f32_buf;

	size_t len = f32buf->size();
	float* f32data = f32buf->data();

	Buffer buf;
	buf.m_sampleRate = 44100;
	buf.Allocate((unsigned)len);
	memcpy(buf.m_data.data(), f32data, sizeof(float)*len);

	std::vector<float> frequencies;
	std::vector<float> dynamics;
	DetectFreqs(buf, frequencies, dynamics, interval);

	float ave = 0.0f;
	float count = 0.0f;
	for (unsigned i = 0; i < (unsigned)frequencies.size(); i++)
	{
		if (frequencies[i] > 55.0f)
		{
			count += 1.0f;
			ave += frequencies[i];
		}
	}
	ave = ave / count;

	frq_data->interval = interval;
	frq_data->key = (double)ave;
	frq_data->data.resize(frequencies.size());
	for (size_t i = 0; i < frequencies.size(); i++)
	{
		frq_data->data[i].freq = (double)frequencies[i];
		frq_data->data[i].dyn = (double)dynamics[i];
	}
}

void* SourceMapCtrlPntCreate(float srcPos, float dstPos, int isVowel)
{
	SourceMapCtrlPnt* pnt = new SourceMapCtrlPnt;
	pnt->srcPos = srcPos;
	pnt->dstPos = dstPos;
	pnt->isVowel = isVowel;
	return pnt;
}

void SourceMapCtrlPntDestroy(void* ptr)
{
	delete (SourceMapCtrlPnt*)ptr;
}

void* PieceCreate(void* ptr_f32buf, void* ptr_frq_data, void* ptr_src_map)
{
	F32Buf* wav = (F32Buf*)ptr_f32buf;
	FrqData* frq_data = (FrqData*)ptr_frq_data;
	PtrArray* src_map = (PtrArray*)ptr_src_map;
	Piece* piece = new Piece;
	piece->src.wav.buf = wav->data();
	piece->src.wav.len = (unsigned)wav->size();
	piece->src.frq = *frq_data;
	piece->srcMap.resize(src_map->size());
	for (size_t i = 0; i < src_map->size(); i++)
	{
		SourceMapCtrlPnt* pnt = (SourceMapCtrlPnt*)(*src_map)[i];
		piece->srcMap[i] = *pnt;
	}
	return piece;
}

void PieceDestroy(void* ptr)
{
	delete (Piece*)ptr;
}

void* GeneralCtrlPntCreate(float value, float dstPos)
{
	GeneralCtrlPnt* pnt = new GeneralCtrlPnt;
	pnt->value = value;
	pnt->dstPos = dstPos;
	return pnt;
}

void GeneralCtrlPntDestroy(void* ptr)
{
	delete (GeneralCtrlPnt*)ptr;
}

void* SentenceDescriptorCreate(void* ptr_pieces, void* ptr_piece_map, void* ptr_freq_map, void* ptr_volume_map)
{
	PtrArray* pieces = (PtrArray*)ptr_pieces;
	PtrArray* piece_map = (PtrArray*)ptr_piece_map;
	PtrArray* freq_map = (PtrArray*)ptr_freq_map;
	PtrArray* volume_map = (PtrArray*)ptr_volume_map;

	SentenceDescriptor* descriptor = new SentenceDescriptor;

	descriptor->pieces.resize(pieces->size());
	for (size_t i = 0; i < pieces->size(); i++)
	{
		Piece* piece = (Piece*)(*pieces)[i];
		descriptor->pieces[i] = *piece;
	}

	descriptor->piece_map.resize(piece_map->size());
	for (size_t i = 0; i < piece_map->size(); i++)
	{
		GeneralCtrlPnt* pnt = (GeneralCtrlPnt*)(*piece_map)[i];
		descriptor->piece_map[i] = *pnt;
	}

	descriptor->freq_map.resize(freq_map->size());
	for (size_t i = 0; i < freq_map->size(); i++)
	{
		GeneralCtrlPnt* pnt = (GeneralCtrlPnt*)(*freq_map)[i];
		descriptor->freq_map[i] = *pnt;
	}

	descriptor->volume_map.resize(volume_map->size());
	for (size_t i = 0; i < volume_map->size(); i++)
	{
		GeneralCtrlPnt* pnt = (GeneralCtrlPnt*)(*volume_map)[i];
		descriptor->volume_map[i] = *pnt;
	}

	return descriptor;
}

void SentenceDescriptorDestroy(void* ptr)
{
	delete (SentenceDescriptor*)ptr;
}

inline void GenerateSentenceX(void* ptr_wavbuf, void* ptr_sentence, bool cuda)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	SentenceDescriptor* sentence = (SentenceDescriptor*)ptr_sentence;

	std::vector<Piece>& pieces = sentence->pieces;
	float falignPos = -pieces[0].srcMap[0].dstPos;

	for (size_t i = 0; i < pieces.size(); i++)
	{
		Piece& piece = pieces[i];
		std::vector<SourceMapCtrlPnt>& srcMap = piece.srcMap;
		for (size_t j = 0; j < srcMap.size(); j++)
		{
			srcMap[j].dstPos += falignPos;
		}
	}

	std::vector<GeneralCtrlPnt>& piece_map = sentence->piece_map;
	std::vector<GeneralCtrlPnt>& freq_map = sentence->freq_map;
	std::vector<GeneralCtrlPnt>& volume_map = sentence->volume_map;

	for (size_t i = 0; i < piece_map.size(); i++)
		piece_map[i].dstPos += falignPos;

	for (size_t i = 0; i < freq_map.size(); i++)
		freq_map[i].dstPos += falignPos;

	for (size_t i = 0; i < volume_map.size(); i++)
		volume_map[i].dstPos += falignPos;

	float flen = freq_map[freq_map.size() - 1].dstPos;

	float rate = wavbuf->m_sampleRate;
	wavbuf->m_alignPos = (unsigned)(falignPos*0.001f*rate + 0.5f);

	size_t len = (size_t)ceilf(flen*0.001f*rate);
	wavbuf->Allocate(1, len);

#ifdef HAVE_CUDA
	if (cuda && HaveCUDA()!=0)
		GenerateSentenceCUDA(sentence, wavbuf->m_data, (unsigned)len);
	else
#endif
		GenerateSentenceCPU(sentence, wavbuf->m_data, (unsigned)len);
}

void GenerateSentence(void* ptr_wavbuf, void* ptr_sentence)
{
	GenerateSentenceX(ptr_wavbuf, ptr_sentence, false);
}

void GenerateSentenceCUDA(void* ptr_wavbuf, void* ptr_sentence)
{
	GenerateSentenceX(ptr_wavbuf, ptr_sentence, true);
}
