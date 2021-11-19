#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	// general
	SCOREDRAFT_API void* PtrArrayCreate(unsigned long long size, const void** ptrs);
	SCOREDRAFT_API void PtrArrayDestroy(void* ptr_arr);

	// F32Buf
	SCOREDRAFT_API void* F32BufCreate(unsigned long long size, float value);
	SCOREDRAFT_API void F32BufDestroy(void* ptr);
	SCOREDRAFT_API float* F32BufData(void* ptr);
	SCOREDRAFT_API int F32BufSize(void* ptr);	
	SCOREDRAFT_API void F32BufToS16(void* ptr, short* dst, float amplitude);
	SCOREDRAFT_API void F32BufFromS16(void* ptr, const short* data, unsigned long long size);
	SCOREDRAFT_API float F32BufMaxValue(void* ptr);
	SCOREDRAFT_API void F32BufMix(void* ptr, void* ptr_lst);

	// WavBuffer
	SCOREDRAFT_API void* WavBufferCreate(float sampleRate, unsigned channelNum, void* ptr_data, unsigned alignPos, float volume, float pan);
	SCOREDRAFT_API void WavBufferDestroy(void *ptr);
	SCOREDRAFT_API float WavBufferGetSampleRate(void* ptr);
	SCOREDRAFT_API unsigned WavBufferGetChannelNum(void *ptr);
	SCOREDRAFT_API unsigned long long WavBufferGetSampleNum(void *ptr);
	SCOREDRAFT_API unsigned WavBufferGetAlignPos(void* ptr);
	SCOREDRAFT_API void WavBufferSetAlignPos(void* ptr, unsigned alignPos);
	SCOREDRAFT_API float WavBufferGetVolume(void* ptr);
	SCOREDRAFT_API void WavBufferSetVolume(void* ptr, float volume);
	SCOREDRAFT_API float WavBufferGetPan(void* ptr);
	SCOREDRAFT_API void WavBufferSetPan(void* ptr, float pan);

	// TrackBuffer
	SCOREDRAFT_API void* TrackBufferCreate(unsigned chn);
	SCOREDRAFT_API void TrackBufferDestroy(void* ptr);
	SCOREDRAFT_API void TrackBufferSetVolume(void* ptr, float volume);
	SCOREDRAFT_API float TrackBufferGetVolume(void* ptr);
	SCOREDRAFT_API void TrackBufferSetPan(void* ptr, float pan);
	SCOREDRAFT_API float TrackBufferGetPan(void* ptr);
	SCOREDRAFT_API unsigned TrackBufferGetNumberOfSamples(void* ptr);
	SCOREDRAFT_API unsigned TrackBufferGetAlignPos(void* ptr);
	SCOREDRAFT_API float TrackBufferGetCursor(void* ptr);
	SCOREDRAFT_API void TrackBufferSetCursor(void* ptr, float cursor);
	SCOREDRAFT_API void TrackBufferMoveCursor(void* ptr, float cursor_delta);
	SCOREDRAFT_API void MixTrackBufferList(void* ptr, void* ptr_list);
	SCOREDRAFT_API void WriteTrackBufferToWav(void* ptr, const char* fn);
	SCOREDRAFT_API void ReadTrackBufferFromWav(void* ptr, const char* fn);
	SCOREDRAFT_API void TrackBufferWriteBlend(void* ptr, void* ptr_wav_buf);

}

#include <memory.h>
#include <math.h>
#include "utils.h"
#include "TrackBuffer.h"

// general
void* PtrArrayCreate(unsigned long long size, const void** ptrs)
{
	PtrArray* ret = new PtrArray(size);
	memcpy(ret->data(), ptrs, sizeof(void*)*size);
	return ret;
}

void PtrArrayDestroy(void* ptr_arr)
{
	PtrArray* arr = (PtrArray*)ptr_arr;
	delete arr;
}

// F32Buf
void* F32BufCreate(unsigned long long size, float value)
{
	return new F32Buf(size, value);
}

void F32BufDestroy(void* ptr)
{
	F32Buf* buf = (F32Buf*)ptr;
	delete buf;
}

float* F32BufData(void* ptr)
{
	F32Buf* buf = (F32Buf*)ptr;
	return buf->data();
}

int F32BufSize(void* ptr)
{
	F32Buf* buf = (F32Buf*)ptr;
	return (int)(buf->size());
}

void F32BufToS16(void* ptr, short* dst, float amplitude)
{
	F32Buf* buf = (F32Buf*)ptr;
	for (size_t i = 0; i < buf->size(); i++)
		dst[i] = (short)((*buf)[i] * 32767.0f*amplitude + 0.5f);
}

void F32BufFromS16(void* ptr, const short* data, unsigned long long size)
{
	F32Buf* buf = (F32Buf*)ptr;
	buf->resize(size);
	for (size_t i = 0; i < size; i++)
	{
		(*buf)[i] = (float)data[i] / 32767.0f;
	}	
}

float F32BufMaxValue(void* ptr)
{
	F32Buf* buf = (F32Buf*)ptr;

	float maxV = 0.0f;
	for (size_t i = 0; i < buf->size(); i++)
	{
		float v = fabsf((*buf)[i]);
		if (v > maxV) maxV = v;
	}

	return maxV;
}

void F32BufMix(void* ptr, void* ptr_lst)
{
	F32Buf* target_buf = (F32Buf*)ptr;
	PtrArray* list = (PtrArray*)ptr_lst;

	unsigned numBufs = (unsigned)list->size();
	size_t maxLen = 0;
	for (unsigned i = 0; i < numBufs; i++)
	{
		F32Buf* buf = (F32Buf*)(*list)[i];
		size_t len = buf->size();

		if (maxLen < len)
			maxLen = (unsigned)len;
	}

	target_buf->resize(maxLen);	
	float* f32Out = target_buf->data();
	memset(f32Out, 0, maxLen * sizeof(float));

	for (unsigned i = 0; i < numBufs; i++)
	{
		F32Buf* buf = (F32Buf*)(*list)[i];
		size_t len = buf->size();
		const float* f32In = buf->data();

		for (unsigned j = 0; j < len; j++)
			f32Out[j] += f32In[j];
	}
}

// WavBuffer
void* WavBufferCreate(float sampleRate, unsigned channelNum, void* ptr_data, unsigned alignPos, float volume, float pan)
{
	WavBuffer* buf = new WavBuffer;
	buf->m_sampleRate = sampleRate;
	buf->m_channelNum = channelNum;
	buf->m_sampleNum = ((F32Buf*)ptr_data)->size() / channelNum;
	buf->p_data = (F32Buf*)ptr_data;
	buf->m_data = &(*buf->p_data)[0];
	buf->m_alignPos = alignPos;
	buf->m_volume = volume;
	buf->m_pan = pan;
	return buf;
}

void WavBufferDestroy(void *ptr)
{
	delete (WavBuffer*)ptr;
}

float WavBufferGetSampleRate(void* ptr)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	return buf->m_sampleRate;
}

unsigned WavBufferGetChannelNum(void *ptr)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	return buf->m_channelNum;
}

unsigned long long WavBufferGetSampleNum(void *ptr)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	return buf->m_sampleNum;
}

unsigned WavBufferGetAlignPos(void* ptr)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	return buf->m_alignPos;
}

void WavBufferSetAlignPos(void* ptr, unsigned alignPos)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	buf->m_alignPos = alignPos;
}

float WavBufferGetVolume(void* ptr)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	return buf->m_volume;
}

void WavBufferSetVolume(void* ptr, float volume)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	buf->m_volume = volume;
}

float WavBufferGetPan(void* ptr)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	return buf->m_pan;
}

void WavBufferSetPan(void* ptr, float pan)
{
	WavBuffer* buf = (WavBuffer*)ptr;
	buf->m_pan = pan;
}

// TrackBuffer
void* TrackBufferCreate(unsigned chn)
{
	return new TrackBuffer(44100, chn);
}

void TrackBufferDestroy(void* ptr)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	delete buffer;
}

void TrackBufferSetVolume(void* ptr, float volume)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	buffer->SetVolume(volume);
}

float TrackBufferGetVolume(void* ptr)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	return buffer->Volume();
}

void TrackBufferSetPan(void* ptr, float pan)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	buffer->SetPan(pan);
}

float TrackBufferGetPan(void* ptr)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	return buffer->Pan();
}

unsigned TrackBufferGetNumberOfSamples(void* ptr)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	return buffer->NumberOfChannels();
}

unsigned TrackBufferGetAlignPos(void* ptr)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	return buffer->AlignPos();
}

float TrackBufferGetCursor(void* ptr)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	return buffer->GetCursor();
}

void TrackBufferSetCursor(void* ptr, float cursor)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	buffer->SetCursor(cursor);
}

void TrackBufferMoveCursor(void* ptr, float cursor_delta)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	buffer->MoveCursor(cursor_delta);
}

void MixTrackBufferList(void* ptr, void* ptr_list)
{
	TrackBuffer* targetBuffer = (TrackBuffer*)ptr;
	PtrArray* list = (PtrArray*)ptr_list;
	targetBuffer->CombineTracks((unsigned)list->size(), (TrackBuffer**)list->data());
}

void WriteTrackBufferToWav(void* ptr, const char* fn)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	WriteToWav(*buffer, fn);
}

void ReadTrackBufferFromWav(void* ptr, const char* fn)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	ReadFromWav(*buffer, fn);
}

void TrackBufferWriteBlend(void* ptr, void* ptr_wav_buf)
{
	TrackBuffer* buffer = (TrackBuffer*)ptr;
	WavBuffer* wavBuf = (WavBuffer*)ptr_wav_buf;
	buffer->WriteBlend(*wavBuf);
}



