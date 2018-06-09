#include "SF2.h"

struct tsf_riffchunk { tsf_fourcc id; tsf_u32 size; };
#define TSF_FourCCEquals(value1, value2) (value1[0] == value2[0] && value1[1] == value2[1] && value1[2] == value2[2] && value1[3] == value2[3])

static TSF_BOOL tsf_riffchunk_read(struct tsf_riffchunk* parent, struct tsf_riffchunk* chunk, struct tsf_stream* stream)
{
	TSF_BOOL IsRiff, IsList;
	if (parent && sizeof(tsf_fourcc) + sizeof(tsf_u32) > parent->size) return TSF_FALSE;
	if (!stream->read(stream->data, &chunk->id, sizeof(tsf_fourcc)) || *chunk->id <= ' ' || *chunk->id >= 'z') return TSF_FALSE;
	if (!stream->read(stream->data, &chunk->size, sizeof(tsf_u32))) return TSF_FALSE;
	if (parent && sizeof(tsf_fourcc) + sizeof(tsf_u32) + chunk->size > parent->size) return TSF_FALSE;
	if (parent) parent->size -= sizeof(tsf_fourcc) + sizeof(tsf_u32) + chunk->size;
	IsRiff = TSF_FourCCEquals(chunk->id, "RIFF"), IsList = TSF_FourCCEquals(chunk->id, "LIST");
	if (IsRiff && parent) return TSF_FALSE; //not allowed
	if (!IsRiff && !IsList) return TSF_TRUE; //custom type without sub type
	if (!stream->read(stream->data, &chunk->id, sizeof(tsf_fourcc)) || *chunk->id <= ' ' || *chunk->id >= 'z') return TSF_FALSE;
	chunk->size -= sizeof(tsf_fourcc);
	return TSF_TRUE;
}

#define TSFR(FIELD) stream->read(stream->data, &i->FIELD, sizeof(i->FIELD));
inline void tsf_hydra_read_phdr(struct tsf_hydra_phdr* i, struct tsf_stream* stream) { TSFR(presetName) TSFR(preset) TSFR(bank) TSFR(presetBagNdx) TSFR(library) TSFR(genre) TSFR(morphology) }
inline void tsf_hydra_read_pbag(struct tsf_hydra_pbag* i, struct tsf_stream* stream) { TSFR(genNdx) TSFR(modNdx) }
inline void tsf_hydra_read_pmod(struct tsf_hydra_pmod* i, struct tsf_stream* stream) { TSFR(modSrcOper) TSFR(modDestOper) TSFR(modAmount) TSFR(modAmtSrcOper) TSFR(modTransOper) }
inline void tsf_hydra_read_pgen(struct tsf_hydra_pgen* i, struct tsf_stream* stream) { TSFR(genOper) TSFR(genAmount) }
inline void tsf_hydra_read_inst(struct tsf_hydra_inst* i, struct tsf_stream* stream) { TSFR(instName) TSFR(instBagNdx) }
inline void tsf_hydra_read_ibag(struct tsf_hydra_ibag* i, struct tsf_stream* stream) { TSFR(instGenNdx) TSFR(instModNdx) }
inline void tsf_hydra_read_imod(struct tsf_hydra_imod* i, struct tsf_stream* stream) { TSFR(modSrcOper) TSFR(modDestOper) TSFR(modAmount) TSFR(modAmtSrcOper) TSFR(modTransOper) }
inline void tsf_hydra_read_igen(struct tsf_hydra_igen* i, struct tsf_stream* stream) { TSFR(genOper) TSFR(genAmount) }
inline void tsf_hydra_read_shdr(struct tsf_hydra_shdr* i, struct tsf_stream* stream) { TSFR(sampleName) TSFR(start) TSFR(end) TSFR(startLoop) TSFR(endLoop) TSFR(sampleRate) TSFR(originalPitch) TSFR(pitchCorrection) TSFR(sampleLink) TSFR(sampleType) }
#undef TSFR


static void tsf_load_samples(F32Samples* samples, struct tsf_riffchunk *chunkSmpl, struct tsf_stream* stream)
{
	// Read sample data into float format buffer.
	float* out; unsigned int samplesTotal, samplesLeft, samplesToRead, samplesToConvert;
	samplesTotal = chunkSmpl->size / sizeof(short);
	samplesLeft = samplesTotal;
	samples->resize(samplesTotal);
	out = samples->data();
	for (; samplesLeft; samplesLeft -= samplesToRead)
	{
		short sampleBuffer[1024], *in = sampleBuffer;;
		samplesToRead = (samplesLeft > 1024 ? 1024 : samplesLeft);
		stream->read(stream->data, sampleBuffer, samplesToRead * sizeof(short));

		// Convert from signed 16-bit to float.
		for (samplesToConvert = samplesToRead; samplesToConvert > 0; --samplesToConvert)
			// If we ever need to compile for big-endian platforms, we'll need to byte-swap here.
			*out++ = (float)(*in++ / 32767.0);
	}
}

Deferred<SF2> LoadSF2(struct tsf_stream* stream)
{
	Deferred<SF2> res;

	struct tsf_riffchunk chunkHead;
	struct tsf_riffchunk chunkList;
	Hydra& hydra = res->hydra;

	if (!tsf_riffchunk_read(TSF_NULL, &chunkHead, stream) || !TSF_FourCCEquals(chunkHead.id, "sfbk"))
	{
		//if (e) *e = TSF_INVALID_NOSF2HEADER;
		return res;
	}

	// Read hydra and locate sample data.
	while (tsf_riffchunk_read(&chunkHead, &chunkList, stream))
	{
		struct tsf_riffchunk chunk;
		if (TSF_FourCCEquals(chunkList.id, "pdta"))
		{
			while (tsf_riffchunk_read(&chunkList, &chunk, stream))
			{
#define HandleChunk(chunkName) (TSF_FourCCEquals(chunk.id, #chunkName) && !(chunk.size % chunkName##SizeInFile)) \
											{ \
						int num = chunk.size / chunkName##SizeInFile, i; \
						hydra.chunkName##s.resize(num);\
						for (i = 0; i < num; ++i) tsf_hydra_read_##chunkName(&hydra.chunkName##s[i], stream); \
								}
				enum
				{
					phdrSizeInFile = 38, pbagSizeInFile = 4, pmodSizeInFile = 10,
					pgenSizeInFile = 4, instSizeInFile = 22, ibagSizeInFile = 4,
					imodSizeInFile = 10, igenSizeInFile = 4, shdrSizeInFile = 46
				};
				if      HandleChunk(phdr) else if HandleChunk(pbag) else if HandleChunk(pmod)
				else if HandleChunk(pgen) else if HandleChunk(inst) else if HandleChunk(ibag)
				else if HandleChunk(imod) else if HandleChunk(igen) else if HandleChunk(shdr)
				else stream->skip(stream->data, chunk.size);
#undef HandleChunk

			}
		}
		else if (TSF_FourCCEquals(chunkList.id, "sdta"))
		{
			while (tsf_riffchunk_read(&chunkList, &chunk, stream))
			{
				if (TSF_FourCCEquals(chunk.id, "smpl"))
				{
					tsf_load_samples(res->fontSamples, &chunk, stream);
				}
				else stream->skip(stream->data, chunk.size);
			}
		}
		else stream->skip(stream->data, chunkList.size);
	}



	return res;

}
