#ifndef _Presets_h
#define _Presets_h

#include "SF2.h"

#include <stdio.h>

enum { TSF_LOOPMODE_NONE, TSF_LOOPMODE_CONTINUOUS, TSF_LOOPMODE_SUSTAIN };

struct tsf_envelope 
{ 
	float delay, attack, hold, decay, sustain, release, keynumToHold, keynumToDecay; 

	void print(FILE* fp, const char* prefix) const
	{
		fprintf(fp, "%s - delay: %f\n", prefix, delay);
		fprintf(fp, "%s - attack: %f\n", prefix, attack);
		fprintf(fp, "%s - hold: %f\n", prefix, hold);
		fprintf(fp, "%s - decay: %f\n", prefix, decay);
		fprintf(fp, "%s - sustain: %f\n", prefix, sustain);
		fprintf(fp, "%s - release: %f\n", prefix, release);
		fprintf(fp, "%s - keynumToHold: %f\n", prefix, keynumToHold);
		fprintf(fp, "%s - keynumToDecay: %f\n", prefix, keynumToDecay);
	}
};

struct tsf_region
{
	int loop_mode;
	unsigned int sample_rate;
	unsigned char lokey, hikey, lovel, hivel;
	unsigned int group, offset, end, loop_start, loop_end;
	int transpose, tune, pitch_keycenter, pitch_keytrack;
	float attenuation, pan;
	tsf_envelope ampenv, modenv;
	int initialFilterQ, initialFilterFc;
	int modEnvToPitch, modEnvToFilterFc, modLfoToFilterFc, modLfoToVolume;
	float delayModLFO;
	int freqModLFO, modLfoToPitch;
	float delayVibLFO;
	int freqVibLFO, vibLfoToPitch;

	void print(FILE* fp) const
	{
		fprintf(fp, "loop_mode: %d\n", loop_mode);
		fprintf(fp, "sample_rate: %u\n", sample_rate);
		fprintf(fp, "offset: %u\n", offset);
		fprintf(fp, "end: %u\n", end);
		fprintf(fp, "loop_start: %u\n", loop_start);
		fprintf(fp, "loop_end: %u\n", loop_end);
		fprintf(fp, "transpose: %d\n", transpose);
		fprintf(fp, "tune: %d\n", tune);
		fprintf(fp, "pitch_keycenter: %d\n", pitch_keycenter);
		fprintf(fp, "pitch_keytrack: %d\n", pitch_keytrack);
		fprintf(fp, "attenuation: %f\n", attenuation);
		fprintf(fp, "pan: %f\n", pan);
		ampenv.print(fp, "ampenv");
		modenv.print(fp, "modenv");
		fprintf(fp, "initialFilterQ: %d\n", initialFilterQ);
		fprintf(fp, "initialFilterFc: %d\n", initialFilterFc);
		fprintf(fp, "modEnvToPitch: %d\n", modEnvToPitch);
		fprintf(fp, "modEnvToFilterFc: %d\n", modEnvToFilterFc);
		fprintf(fp, "modLfoToFilterFc: %d\n", modLfoToFilterFc);
		fprintf(fp, "modLfoToVolume: %d\n", modLfoToVolume);
		fprintf(fp, "delayModLFO: %f\n", delayModLFO);
		fprintf(fp, "freqModLFO: %d\n", freqModLFO);
		fprintf(fp, "modLfoToPitch: %d\n", modLfoToPitch);
		fprintf(fp, "delayVibLFO: %f\n", delayVibLFO);
		fprintf(fp, "freqVibLFO: %d\n", freqVibLFO);
		fprintf(fp, "vibLfoToPitch: %d\n", vibLfoToPitch);	
	}
};

struct tsf_preset
{
	tsf_char20 presetName;
	tsf_u16 preset, bank;
	std::vector<tsf_region> regions;
};

typedef std::vector<tsf_preset> Presets;

void LoadPresets(SF2& sf2, Presets& presets);


#endif

