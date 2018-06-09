#ifndef _Synth_h
#define _Synth_h

#include <vector>

struct LowPassState
{
	double z1, z2;
};

struct NoteState
{
	double sourceSamplePosition;
	LowPassState lowPass;
};

struct LowPassCtrlPnt
{
	char active;
	double a0, a1, b1, b2;
};

struct SynthCtrlPnt
{
	char looping;
	float gainMono;
	double pitchRatio;

	LowPassCtrlPnt lowPass;	
};

enum OutputMode
{
	// Two channels with single left/right samples one after another
	STEREO_INTERLEAVED,
	// Two channels with all samples for the left channel first then right
	STEREO_UNWEAVED,
	// A single channel (stereo instruments are mixed into center)
	MONO,
};

struct SynthCtrl
{
	OutputMode outputmode;
	unsigned loopStart, loopEnd;
	unsigned end;
	float panFactorLeft, panFactorRight;
	unsigned effect_sample_block;
	std::vector<SynthCtrlPnt> controlPnts;
};

void Synth(const float* input, float* outputBuffer, unsigned numSamples, NoteState& noteState, const SynthCtrl& control);

#endif
