#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define SCOREDRAFT_API __declspec(dllexport)
#else
#define SCOREDRAFT_API 
#endif

extern "C"
{
	// SF2Bank
	SCOREDRAFT_API void* SF2BankCreate(const char* filename);
	SCOREDRAFT_API void SF2BankDestroy(void* ptr);
	SCOREDRAFT_API unsigned long long SF2BankGetNumberPresets(void* ptr);	
	SCOREDRAFT_API const char* SF2BankGetPresetName(void* ptr, int i);
	SCOREDRAFT_API int SF2BankGetPresetBankNum(void* ptr, int i);
	SCOREDRAFT_API int SF2BankGetPresetNumber(void* ptr, int i);

	// SF2Tone
	SCOREDRAFT_API void* SF2ToneCreate(void* ptr_bank, unsigned preset_index);
	SCOREDRAFT_API void SF2ToneDestroy(void* ptr);

	// SF2Synth
	SCOREDRAFT_API void SF2SynthNote(void* ptr_wavbuf, void* ptr_tone, float key, float vel, unsigned numSamples, unsigned outputmode, float global_gain_db);
}

#include "SF2Synth.h"
#include <WavBuffer.h>

// SF2Bank
struct SF2Bank
{
	std::shared_ptr<F32Samples> font_samples;
	Presets presets;
};

void* SF2BankCreate(const char* filename)
{
	SF2Bank* bank = new SF2Bank;
	SF2 sf2;
	LoadSF2Filename(filename, sf2);
	bank->font_samples = sf2.fontSamples;
	LoadPresets(sf2, bank->presets);
	return bank;
}

void SF2BankDestroy(void* ptr)
{
	SF2Bank* bank = (SF2Bank*)ptr;
	delete bank;
}

unsigned long long SF2BankGetNumberPresets(void* ptr)
{
	SF2Bank* bank = (SF2Bank*)ptr;
	return bank->presets.size();
}

const char* SF2BankGetPresetName(void* ptr, int i)
{
	SF2Bank* bank = (SF2Bank*)ptr;
	return bank->presets[i].presetName;
}

int SF2BankGetPresetBankNum(void* ptr, int i)
{
	SF2Bank* bank = (SF2Bank*)ptr;
	return bank->presets[i].bank;
}

int SF2BankGetPresetNumber(void* ptr, int i)
{
	SF2Bank* bank = (SF2Bank*)ptr;
	return bank->presets[i].preset;
}

// SF2Tone
struct SF2Tone
{
	F32Samples* input;
	tsf_preset* preset;
};

void* SF2ToneCreate(void* ptr_bank, unsigned preset_index)
{
	SF2Bank* bank = (SF2Bank*)ptr_bank;
	SF2Tone* tone = new SF2Tone;
	tone->input = bank->font_samples.get();
	tone->preset = &bank->presets[preset_index];	
	return tone;
}

void SF2ToneDestroy(void* ptr)
{
	SF2Tone* tone = (SF2Tone*)ptr;
	delete tone;
}

// SF2Synth
void SF2SynthNote(void* ptr_wavbuf, void* ptr_tone, float key, float vel, unsigned numSamples, unsigned outputmode, float global_gain_db)
{
	WavBuffer* wavbuf = (WavBuffer*)ptr_wavbuf;
	wavbuf->m_channelNum = outputmode < 2 ? 2 : 1;	
	SF2Tone* tone = (SF2Tone*)ptr_tone;
	float samplerate = wavbuf->m_sampleRate;
	SF2Synth(*wavbuf->p_data, *tone->input, *tone->preset, key, vel, numSamples, (OutputMode)outputmode, samplerate, global_gain_db);
	wavbuf->m_sampleNum = wavbuf->p_data->size()/ wavbuf->m_channelNum;
	wavbuf->m_data = wavbuf->p_data->data();		
}