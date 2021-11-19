#ifndef _SF2Synth_h
#define _SF2Synth_h

#include "SF2.h"
#include "Presets.h"
#include "Synth.h"

#include <memory>

void SF2Synth(F32Samples& output, F32Samples& input, tsf_preset& preset, float key, float vel, unsigned& numSamples,
	OutputMode outputmode = STEREO_INTERLEAVED, float samplerate = 44100.0f, float global_gain_db = 0.0f);


#endif 

