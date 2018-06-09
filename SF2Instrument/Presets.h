#ifndef _Presets_h
#define _Presets_h

#include "SF2.h"

enum { TSF_LOOPMODE_NONE, TSF_LOOPMODE_CONTINUOUS, TSF_LOOPMODE_SUSTAIN };

struct tsf_envelope { float delay, attack, hold, decay, sustain, release, keynumToHold, keynumToDecay; };

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
};

struct tsf_preset
{
	tsf_char20 presetName;
	tsf_u16 preset, bank;
	std::vector<tsf_region> regions;
};

typedef std::vector<tsf_preset> Presets;
typedef Deferred<Presets> Presets_deferred;

Presets_deferred LoadPresets(SF2* sf2);


#endif

