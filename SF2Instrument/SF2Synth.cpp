#include "SF2Synth.h"

struct LowPass
{
	char active;
	double QInv;
	double a0, a1, b1, b2;
};

struct tsf_voice_envelope { float level, slope; int samplesUntilNextSegment; short segment, midiVelocity; struct tsf_envelope parameters; TSF_BOOL segmentIsExponential, isAmpEnv; };
struct tsf_voice_lfo { int samplesUntil; float level, delta; };

struct Voice
{
	NoteState ns;
	tsf_region* region;
	double pitchInputTimecents, pitchOutputFactor;
	float  noteGainDB, panFactorLeft, panFactorRight;
	unsigned int loopStart, loopEnd;
	tsf_voice_envelope ampenv, modenv;
	LowPass lowpass;
	tsf_voice_lfo modlfo, viblfo;
};


#if !defined(TSF_POW) || !defined(TSF_POWF) || !defined(TSF_EXPF) || !defined(TSF_LOG) || !defined(TSF_TAN) || !defined(TSF_LOG10) || !defined(TSF_SQRT)
#  include <math.h>
#  if !defined(__cplusplus) && !defined(NAN) && !defined(powf) && !defined(expf) && !defined(sqrtf)
#    define powf (float)pow // deal with old math.h
#    define expf (float)exp // files that come without
#    define sqrtf (float)sqrt // powf, expf and sqrtf
#  endif
#  define TSF_POW     pow
#  define TSF_POWF    powf
#  define TSF_EXPF    expf
#  define TSF_LOG     log
#  define TSF_TAN     tan
#  define TSF_LOG10   log10
#  define TSF_SQRTF   sqrtf
#endif

static float tsf_gainToDecibels(float gain) { return (gain <= .00001f ? -100.f : (float)(20.0 * TSF_LOG10(gain))); }
static double tsf_timecents2Secsd(double timecents) { return TSF_POW(2.0, timecents / 1200.0); }
static float tsf_timecents2Secsf(float timecents) { return TSF_POWF(2.0f, timecents / 1200.0f); }
static float tsf_cents2Hertz(float cents) { return 8.176f * TSF_POWF(2.0f, cents / 1200.0f); }
static float tsf_decibelsToGain(float db) { return (db > -100.f ? TSF_POWF(10.0f, db * 0.05f) : 0); }

enum { TSF_SEGMENT_NONE, TSF_SEGMENT_DELAY, TSF_SEGMENT_ATTACK, TSF_SEGMENT_HOLD, TSF_SEGMENT_DECAY, TSF_SEGMENT_SUSTAIN, TSF_SEGMENT_RELEASE, TSF_SEGMENT_DONE };
#define TSF_FASTRELEASETIME 0.01f

static void tsf_voice_envelope_nextsegment(struct tsf_voice_envelope* e, short active_segment, float outSampleRate)
{
	switch (active_segment)
	{
	case TSF_SEGMENT_NONE:
		e->samplesUntilNextSegment = (int)(e->parameters.delay * outSampleRate);
		if (e->samplesUntilNextSegment > 0)
		{
			e->segment = TSF_SEGMENT_DELAY;
			e->segmentIsExponential = TSF_FALSE;
			e->level = 0.0;
			e->slope = 0.0;
			return;
		}
	case TSF_SEGMENT_DELAY:
		e->samplesUntilNextSegment = (int)(e->parameters.attack * outSampleRate);
		if (e->samplesUntilNextSegment > 0)
		{
			if (!e->isAmpEnv)
			{
				//mod env attack duration scales with velocity (velocity of 1 is full duration, max velocity is 0.125 times duration)
				e->samplesUntilNextSegment = (int)(e->parameters.attack * ((145 - e->midiVelocity) / 144.0f) * outSampleRate);
			}
			e->segment = TSF_SEGMENT_ATTACK;
			e->segmentIsExponential = TSF_FALSE;
			e->level = 0.0f;
			e->slope = 1.0f / e->samplesUntilNextSegment;
			return;
		}
	case TSF_SEGMENT_ATTACK:
		e->samplesUntilNextSegment = (int)(e->parameters.hold * outSampleRate);
		if (e->samplesUntilNextSegment > 0)
		{
			e->segment = TSF_SEGMENT_HOLD;
			e->segmentIsExponential = TSF_FALSE;
			e->level = 1.0f;
			e->slope = 0.0f;
			return;
		}
	case TSF_SEGMENT_HOLD:
		e->samplesUntilNextSegment = (int)(e->parameters.decay * outSampleRate);
		if (e->samplesUntilNextSegment > 0)
		{
			e->segment = TSF_SEGMENT_DECAY;
			e->level = 1.0f;
			if (e->isAmpEnv)
			{
				// I don't truly understand this; just following what LinuxSampler does.
				float mysterySlope = -9.226f / e->samplesUntilNextSegment;
				e->slope = TSF_EXPF(mysterySlope);
				e->segmentIsExponential = TSF_TRUE;
				if (e->parameters.sustain > 0.0f)
				{
					// Again, this is following LinuxSampler's example, which is similar to
					// SF2-style decay, where "decay" specifies the time it would take to
					// get to zero, not to the sustain level.  The SFZ spec is not that
					// specific about what "decay" means, so perhaps it's really supposed
					// to specify the time to reach the sustain level.
					e->samplesUntilNextSegment = (int)(TSF_LOG(e->parameters.sustain) / mysterySlope);
				}
			}
			else
			{
				e->slope = -1.0f / e->samplesUntilNextSegment;
				e->samplesUntilNextSegment = (int)(e->parameters.decay * (1.0f - e->parameters.sustain) * outSampleRate);
				e->segmentIsExponential = TSF_FALSE;
			}
			return;
		}
	case TSF_SEGMENT_DECAY:
		e->segment = TSF_SEGMENT_SUSTAIN;
		e->level = e->parameters.sustain;
		e->slope = 0.0f;
		e->samplesUntilNextSegment = 0x7FFFFFFF;
		e->segmentIsExponential = TSF_FALSE;
		return;
	case TSF_SEGMENT_SUSTAIN:
		e->segment = TSF_SEGMENT_RELEASE;
		e->samplesUntilNextSegment = (int)((e->parameters.release <= 0 ? TSF_FASTRELEASETIME : e->parameters.release) * outSampleRate);
		if (e->isAmpEnv)
		{
			// I don't truly understand this; just following what LinuxSampler does.
			float mysterySlope = -9.226f / e->samplesUntilNextSegment;
			e->slope = TSF_EXPF(mysterySlope);
			e->segmentIsExponential = TSF_TRUE;
		}
		else
		{
			e->slope = -e->level / e->samplesUntilNextSegment;
			e->segmentIsExponential = TSF_FALSE;
		}
		return;
	case TSF_SEGMENT_RELEASE:
	default:
		e->segment = TSF_SEGMENT_DONE;
		e->segmentIsExponential = TSF_FALSE;
		e->level = e->slope = 0.0f;
		e->samplesUntilNextSegment = 0x7FFFFFF;
	}
}

static void tsf_voice_envelope_setup(struct tsf_voice_envelope* e, struct tsf_envelope* new_parameters, float midiNoteNumber, short midiVelocity, TSF_BOOL isAmpEnv, float outSampleRate)
{
	e->parameters = *new_parameters;
	if (e->parameters.keynumToHold!=0.0f)
	{
		e->parameters.hold += e->parameters.keynumToHold * (60.0f - midiNoteNumber);
		e->parameters.hold = (e->parameters.hold < -10000.0f ? 0.0f : tsf_timecents2Secsf(e->parameters.hold));
	}
	if (e->parameters.keynumToDecay != 0.0f)
	{
		e->parameters.decay += e->parameters.keynumToDecay * (60.0f - midiNoteNumber);
		e->parameters.decay = (e->parameters.decay < -10000.0f ? 0.0f : tsf_timecents2Secsf(e->parameters.decay));
	}
	e->midiVelocity = midiVelocity;
	e->isAmpEnv = isAmpEnv;
	tsf_voice_envelope_nextsegment(e, TSF_SEGMENT_NONE, outSampleRate);
}

static void tsf_voice_lowpass_setup(LowPass* e, float Fc)
{
	// Lowpass filter from http://www.earlevel.com/main/2012/11/26/biquad-c-source-code/
	double  K = TSF_TAN(TSF_PI * Fc), KK = K * K;
	double  norm = 1 / (1 + K * e->QInv + KK);
	e->a0 = KK * norm;
	e->a1 = 2 * e->a0;
	e->b1 = 2 * (KK - 1) * norm;
	e->b2 = (1 - K * e->QInv + KK) * norm;
}


static void tsf_voice_lfo_setup(struct tsf_voice_lfo* e, float delay, int freqCents, float outSampleRate)
{
	e->samplesUntil = (int)(delay * outSampleRate);
	e->delta = (4.0f * tsf_cents2Hertz((float)freqCents) / outSampleRate);
	e->level = 0;
}

static void tsf_voice_envelope_process(struct tsf_voice_envelope* e, int numSamples, float outSampleRate)
{
	if (e->slope)
	{
		if (e->segmentIsExponential) e->level *= TSF_POWF(e->slope, (float)numSamples);
		else e->level += (e->slope * numSamples);
	}
	if ((e->samplesUntilNextSegment -= numSamples) <= 0)
		tsf_voice_envelope_nextsegment(e, e->segment, outSampleRate);
}


static void tsf_voice_lfo_process(struct tsf_voice_lfo* e, int blockSamples)
{
	if (e->samplesUntil > blockSamples) { e->samplesUntil -= blockSamples; return; }
	e->level += e->delta * blockSamples;
	if (e->level >  1.0f) { e->delta = -e->delta; e->level = 2.0f - e->level; }
	else if (e->level < -1.0f) { e->delta = -e->delta; e->level = -2.0f - e->level; }
}

// The lower this block size is the more accurate the effects are.
// Increasing the value significantly lowers the CPU usage of the voice rendering.
// If LFO affects the low-pass filter it can be hearable even as low as 8.
#ifndef TSF_RENDER_EFFECTSAMPLEBLOCK
#define TSF_RENDER_EFFECTSAMPLEBLOCK 64
#endif


static F32Samples_deferred SynthVoice(F32Samples& input, unsigned& numSamples, Voice& voice, OutputMode outputmode, float samplerate)
{
	tsf_region* region = voice.region;

	TSF_BOOL updateModEnv = (region->modEnvToPitch != 0 || region->modEnvToFilterFc != 0);
	TSF_BOOL updateModLFO = (voice.modlfo.delta != 0.0f && (region->modLfoToPitch != 0 || region->modLfoToFilterFc != 0 || region->modLfoToVolume != 0));
	TSF_BOOL updateVibLFO = (voice.viblfo.delta != 0.0f && (region->vibLfoToPitch != 0));
	TSF_BOOL isLooping = (voice.loopStart < voice.loopEnd);

	unsigned tmpLoopStart = voice.loopStart;
	unsigned tmpLoopEnd = voice.loopEnd;

	double tmpSampleEndDbl = (double)region->end;
	double tmpLoopEndDbl = (double)tmpLoopEnd + 1.0f;
	double tmpSourceSamplePosition = voice.ns.sourceSamplePosition;

	LowPass tmpLowpass = voice.lowpass;
	TSF_BOOL dynamicLowpass = (region->modLfoToFilterFc != 0 || region->modEnvToFilterFc != 0);
	float tmpSampleRate, tmpInitialFilterFc, tmpModLfoToFilterFc, tmpModEnvToFilterFc;

	TSF_BOOL dynamicPitchRatio = (region->modLfoToPitch != 0 || region->modEnvToPitch != 0 || region->vibLfoToPitch != 0);
	double pitchRatio;
	float tmpModLfoToPitch, tmpVibLfoToPitch, tmpModEnvToPitch;

	TSF_BOOL dynamicGain = (region->modLfoToVolume != 0);
	float noteGain = 0, tmpModLfoToVolume;

	if (dynamicLowpass) tmpSampleRate = samplerate, tmpInitialFilterFc = (float)region->initialFilterFc, tmpModLfoToFilterFc = (float)region->modLfoToFilterFc, tmpModEnvToFilterFc = (float)region->modEnvToFilterFc;
	else tmpSampleRate = 0, tmpInitialFilterFc = 0, tmpModLfoToFilterFc = 0, tmpModEnvToFilterFc = 0;
	if (dynamicPitchRatio) pitchRatio = 0, tmpModLfoToPitch = (float)region->modLfoToPitch, tmpVibLfoToPitch = (float)region->vibLfoToPitch, tmpModEnvToPitch = (float)region->modEnvToPitch;
	else pitchRatio = tsf_timecents2Secsd(voice.pitchInputTimecents) * voice.pitchOutputFactor, tmpModLfoToPitch = 0, tmpVibLfoToPitch = 0, tmpModEnvToPitch = 0;

	if (dynamicGain) tmpModLfoToVolume = (float)region->modLfoToVolume * 0.1f;
	else noteGain = tsf_decibelsToGain(voice.noteGainDB), tmpModLfoToVolume = 0;

	SynthCtrl control;
	control.outputmode = outputmode;
	control.loopStart = tmpLoopStart;
	control.loopEnd = tmpLoopEnd;
	control.end = region->end;
	control.panFactorLeft = voice.panFactorLeft;
	control.panFactorRight = voice.panFactorRight;
	control.effect_sample_block = TSF_RENDER_EFFECTSAMPLEBLOCK;
	
	unsigned countSamples = 0;

	while (true)
	{
		float gainMono;
		int blockSamples = TSF_RENDER_EFFECTSAMPLEBLOCK;
		countSamples += blockSamples;

		if (countSamples >= numSamples && voice.ampenv.segment<TSF_SEGMENT_RELEASE)
		{
			tsf_voice_envelope_nextsegment(&voice.ampenv, TSF_SEGMENT_SUSTAIN, samplerate);
			tsf_voice_envelope_nextsegment(&voice.modenv, TSF_SEGMENT_SUSTAIN, samplerate);
			if (voice.region->loop_mode == TSF_LOOPMODE_SUSTAIN)
				// Continue playing, but stop looping.
				isLooping = false;
		}

		if (dynamicLowpass)
		{
			float fres = tmpInitialFilterFc + voice.modlfo.level * tmpModLfoToFilterFc + voice.modenv.level * tmpModEnvToFilterFc;
			tmpLowpass.active = (fres <= 13500.0f);
			if (tmpLowpass.active) tsf_voice_lowpass_setup(&tmpLowpass, tsf_cents2Hertz(fres) / tmpSampleRate);
		}

		if (dynamicPitchRatio)
			pitchRatio = tsf_timecents2Secsd(voice.pitchInputTimecents + (voice.modlfo.level * tmpModLfoToPitch + voice.viblfo.level * tmpVibLfoToPitch + voice.modenv.level * tmpModEnvToPitch)) *  voice.pitchOutputFactor;

		if (dynamicGain)
			noteGain = tsf_decibelsToGain(voice.noteGainDB + (voice.modlfo.level * tmpModLfoToVolume));

		gainMono = noteGain * voice.ampenv.level;

		// Update EG.
		tsf_voice_envelope_process(&voice.ampenv, blockSamples, samplerate);
		if (updateModEnv) tsf_voice_envelope_process(&voice.modenv, blockSamples, samplerate);

		// Update LFOs.
		if (updateModLFO) tsf_voice_lfo_process(&voice.modlfo, blockSamples);
		if (updateVibLFO) tsf_voice_lfo_process(&voice.viblfo, blockSamples);

		SynthCtrlPnt ctrlPnt;
		ctrlPnt.looping = isLooping;
		ctrlPnt.gainMono = gainMono;
		ctrlPnt.pitchRatio = pitchRatio;
		ctrlPnt.lowPass.active = tmpLowpass.active;
		ctrlPnt.lowPass.a0 = tmpLowpass.a0;
		ctrlPnt.lowPass.a1 = tmpLowpass.a1;
		ctrlPnt.lowPass.b1 = tmpLowpass.b1;
		ctrlPnt.lowPass.b2 = tmpLowpass.b2;
		control.controlPnts.push_back(ctrlPnt);

		tmpSourceSamplePosition += pitchRatio*(float)blockSamples;
		while (tmpSourceSamplePosition >= tmpLoopEndDbl && isLooping) 
			tmpSourceSamplePosition -= (tmpLoopEnd - tmpLoopStart + 1.0f);

		if (tmpSourceSamplePosition >= tmpSampleEndDbl || voice.ampenv.segment == TSF_SEGMENT_DONE)
			break;

	}
	
	numSamples = countSamples;
	unsigned chn = outputmode == MONO ? 1 : 2;

	F32Samples_deferred outBuf;
	outBuf->resize(countSamples*chn);

	memset(outBuf->data(), 0, sizeof(float)* countSamples*chn);
	Synth(input.data(), outBuf->data(), countSamples, voice.ns, control);

	return outBuf;
}

F32Samples_deferred  SF2Synth(F32Samples& input, tsf_preset& preset, float key, float vel,
	unsigned& numSamples, OutputMode outputmode, float samplerate, float global_gain_db)
{
	int midiVelocity = (int)(vel * 127);
	std::vector<tsf_region>::iterator region, regionEnd;
	std::vector<Voice> voices;

	int iKey = (int)(key + 0.5f);

	for (region = preset.regions.begin(), regionEnd = region + preset.regions.size();
		region != regionEnd; region++)
	{
		if (iKey < region->lokey || iKey > region->hikey || midiVelocity < region->lovel || midiVelocity > region->hivel) continue;

		Voice voice;
		voice.region = region._Ptr;
		voice.noteGainDB = global_gain_db - region->attenuation - tsf_gainToDecibels(1.0f / vel);

		{
			double note = (double)key + (double)region->transpose + (double)region->tune / 100.0;
			double adjustedPitch = (double)region->pitch_keycenter + (note - (double)region->pitch_keycenter)* ((double)region->pitch_keytrack / 100.0);
			voice.pitchInputTimecents = adjustedPitch * 100.0;
			voice.pitchOutputFactor = (double)region->sample_rate / (tsf_timecents2Secsd((double)region->pitch_keycenter * 100.0) * (double)samplerate);
		}
		// The SFZ spec is silent about the pan curve, but a 3dB pan law seems common. This sqrt() curve matches what Dimension LE does; Alchemy Free seems closer to sin(adjustedPan * pi/2).
		voice.panFactorLeft = TSF_SQRTF(0.5f - region->pan);
		voice.panFactorRight = TSF_SQRTF(0.5f + region->pan);

		// Offset/end.
		voice.ns.sourceSamplePosition = region->offset;

		// Loop.
		bool doLoop = (region->loop_mode != TSF_LOOPMODE_NONE && region->loop_start < region->loop_end);
		voice.loopStart = (doLoop ? region->loop_start : 0);
		voice.loopEnd = (doLoop ? region->loop_end : 0);

		// Setup envelopes.
		tsf_voice_envelope_setup(&voice.ampenv, &region->ampenv, key, midiVelocity, TSF_TRUE, samplerate);
		tsf_voice_envelope_setup(&voice.modenv, &region->modenv, key, midiVelocity, TSF_FALSE, samplerate);

		// Setup lowpass filter.
		float filterQDB = region->initialFilterQ / 10.0f;
		voice.lowpass.QInv = 1.0f / TSF_POW(10.0f, (filterQDB / 20.0f));
		voice.ns.lowPass.z1 = voice.ns.lowPass.z2 = 0;
		voice.lowpass.active = (region->initialFilterFc <= 13500);
		if (voice.lowpass.active)
			tsf_voice_lowpass_setup(&voice.lowpass, tsf_cents2Hertz((float)region->initialFilterFc) / samplerate);


		// Setup LFO filters.
		tsf_voice_lfo_setup(&voice.modlfo, region->delayModLFO, region->freqModLFO, samplerate);
		tsf_voice_lfo_setup(&voice.viblfo, region->delayVibLFO, region->freqVibLFO, samplerate);


		voices.push_back(voice);

	}

	if (voices.size() < 1) 
		return F32Samples_deferred();
	else if (voices.size() < 2)
		return SynthVoice(input, numSamples, voices[0], outputmode, samplerate);
	else
	{
		std::vector<F32Samples_deferred> results;
		unsigned maxNumSamples = 0;
		for (unsigned i = 0; i < voices.size(); i++)
		{
			unsigned countSamples = numSamples;
			results.push_back(SynthVoice(input, countSamples, voices[i], outputmode, samplerate));
			if (countSamples > maxNumSamples)
				maxNumSamples = countSamples;
		}
		unsigned chn = outputmode == MONO ? 1 : 2;

		F32Samples_deferred outBuf;
		outBuf->resize(maxNumSamples*chn);
		memset(outBuf->data(), 0, sizeof(float)* maxNumSamples*chn);

		for (unsigned i = 0; i < voices.size(); i++)
		{
			F32Samples_deferred result = results[i];
			for (unsigned j = 0; j < result->size(); j++)
				(*outBuf)[j] += (*result)[j];
		}

		numSamples = maxNumSamples;
		return outBuf;
	}

}


