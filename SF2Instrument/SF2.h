#ifndef __SF2_h
#define __SF2_h

#include <vector>
#include "Deferred.h"

#if !defined(TSF_MALLOC) || !defined(TSF_FREE) || !defined(TSF_REALLOC)
#  include <stdlib.h>
#  define TSF_MALLOC  malloc
#  define TSF_FREE    free
#  define TSF_REALLOC realloc
#endif

#if !defined(TSF_MEMCPY) || !defined(TSF_MEMSET)
#  include <string.h>
#  define TSF_MEMCPY  memcpy
#  define TSF_MEMSET  memset
#endif

#ifndef TSF_NO_STDIO
#  include <stdio.h>
#endif

#define TSF_TRUE 1
#define TSF_FALSE 0
#define TSF_BOOL char
#define TSF_PI 3.14159265358979323846264338327950288
#define TSF_NULL 0

typedef char tsf_fourcc[4];
typedef signed char tsf_s8;
typedef unsigned char tsf_u8;
typedef unsigned short tsf_u16;
typedef signed short tsf_s16;
typedef unsigned int tsf_u32;
typedef char tsf_char20[20];

union tsf_hydra_genamount { struct { tsf_u8 lo, hi; } range; tsf_s16 shortAmount; tsf_u16 wordAmount; };
struct tsf_hydra_phdr { tsf_char20 presetName; tsf_u16 preset, bank, presetBagNdx; tsf_u32 library, genre, morphology; };
struct tsf_hydra_pbag { tsf_u16 genNdx, modNdx; };
struct tsf_hydra_pmod { tsf_u16 modSrcOper, modDestOper; tsf_s16 modAmount; tsf_u16 modAmtSrcOper, modTransOper; };
struct tsf_hydra_pgen { tsf_u16 genOper; union tsf_hydra_genamount genAmount; };
struct tsf_hydra_inst { tsf_char20 instName; tsf_u16 instBagNdx; };
struct tsf_hydra_ibag { tsf_u16 instGenNdx, instModNdx; };
struct tsf_hydra_imod { tsf_u16 modSrcOper, modDestOper; tsf_s16 modAmount; tsf_u16 modAmtSrcOper, modTransOper; };
struct tsf_hydra_igen { tsf_u16 genOper; union tsf_hydra_genamount genAmount; };
struct tsf_hydra_shdr { tsf_char20 sampleName; tsf_u32 start, end, startLoop, endLoop, sampleRate; tsf_u8 originalPitch; tsf_s8 pitchCorrection; tsf_u16 sampleLink, sampleType; };


// Stream structure for the generic loading
struct tsf_stream
{
	// Custom data given to the functions as the first parameter
	void* data;

	// Function pointer will be called to read 'size' bytes into ptr (returns number of read bytes)
	int(*read)(void* data, void* ptr, unsigned int size);

	// Function pointer will be called to skip ahead over 'count' bytes (returns 1 on success, 0 on error)
	int(*skip)(void* data, unsigned int count);
};

struct Hydra
{
	std::vector<tsf_hydra_phdr> phdrs; std::vector<tsf_hydra_pbag> pbags; std::vector<tsf_hydra_pmod> pmods;
	std::vector<tsf_hydra_pgen> pgens; std::vector<tsf_hydra_inst> insts; std::vector<tsf_hydra_ibag> ibags;
	std::vector<tsf_hydra_imod> imods; std::vector<tsf_hydra_igen> igens; std::vector<tsf_hydra_shdr> shdrs;
};

typedef std::vector<float> F32Samples;
typedef Deferred<F32Samples> F32Samples_deferred;

struct SF2
{
	Hydra hydra;
	F32Samples_deferred fontSamples;
};

Deferred<SF2> LoadSF2(struct tsf_stream* stream);

#ifndef TSF_NO_STDIO
inline int tsf_stream_stdio_read(FILE* f, void* ptr, unsigned int size) { return (int)fread(ptr, 1, size, f); }
inline int tsf_stream_stdio_skip(FILE* f, unsigned int count) { return !fseek(f, count, SEEK_CUR); }
inline Deferred<SF2> LoadSF2Filename(const char* filename)
{
	struct tsf_stream stream = { TSF_NULL, (int(*)(void*, void*, unsigned int))&tsf_stream_stdio_read, (int(*)(void*, unsigned int))&tsf_stream_stdio_skip };
#if __STDC_WANT_SECURE_LIB__
	FILE* f = TSF_NULL; fopen_s(&f, filename, "rb");
#else
	FILE* f = fopen(filename, "rb");
#endif
	if (!f)
	{
		//if (e) *e = TSF_FILENOTFOUND;
		return Deferred<SF2>();
	}
	stream.data = f;
	Deferred<SF2> res = LoadSF2(&stream);
	fclose(f);
	return res;
}
#endif

struct tsf_stream_memory { const char* buffer; unsigned int total, pos; };
inline int tsf_stream_memory_read(struct tsf_stream_memory* m, void* ptr, unsigned int size) { if (size > m->total - m->pos) size = m->total - m->pos; TSF_MEMCPY(ptr, m->buffer + m->pos, size); m->pos += size; return size; }
inline int tsf_stream_memory_skip(struct tsf_stream_memory* m, unsigned int count) { if (m->pos + count > m->total) return 0; m->pos += count; return 1; }

inline Deferred<SF2> LoadSF2Memory(const void* buffer, int size)
{
	struct tsf_stream stream = { TSF_NULL, (int(*)(void*, void*, unsigned int))&tsf_stream_memory_read, (int(*)(void*, unsigned int))&tsf_stream_memory_skip };
	struct tsf_stream_memory f = { 0, 0, 0 };
	f.buffer = (const char*)buffer;
	f.total = size;
	stream.data = &f;
	return LoadSF2(&stream);
}


#endif
