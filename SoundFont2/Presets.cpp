#include "Presets.h"


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


static void tsf_region_clear(struct tsf_region* i, TSF_BOOL for_relative)
{
	TSF_MEMSET(i, 0, sizeof(struct tsf_region));
	i->hikey = i->hivel = 127;
	i->pitch_keycenter = 60; // C4
	if (for_relative) return;

	i->pitch_keytrack = 100;

	i->pitch_keycenter = -1;

	// SF2 defaults in timecents.
	i->ampenv.delay = i->ampenv.attack = i->ampenv.hold = i->ampenv.decay = i->ampenv.release = -12000.0f;
	i->modenv.delay = i->modenv.attack = i->modenv.hold = i->modenv.decay = i->modenv.release = -12000.0f;

	i->initialFilterFc = 13500;

	i->delayModLFO = -12000.0f;
	i->delayVibLFO = -12000.0f;
}

static float tsf_timecents2Secsf(float timecents) { return TSF_POWF(2.0f, timecents / 1200.0f); }
static float tsf_decibelsToGain(float db) { return (db > -100.f ? TSF_POWF(10.0f, db * 0.05f) : 0); }

static void tsf_region_envtosecs(struct tsf_envelope* p, TSF_BOOL sustainIsGain)
{
	// EG times need to be converted from timecents to seconds.
	// Pin very short EG segments.  Timecents don't get to zero, and our EG is
	// happier with zero values.
	p->delay = (p->delay   < -11950.0f ? 0.0f : tsf_timecents2Secsf(p->delay));
	p->attack = (p->attack  < -11950.0f ? 0.0f : tsf_timecents2Secsf(p->attack));
	p->release = (p->release < -11950.0f ? 0.0f : tsf_timecents2Secsf(p->release));

	// If we have dynamic hold or decay times depending on key number we need
	// to keep the values in timecents so we can calculate it during startNote
	if (!p->keynumToHold)  p->hold = (p->hold  < -11950.0f ? 0.0f : tsf_timecents2Secsf(p->hold));
	if (!p->keynumToDecay) p->decay = (p->decay < -11950.0f ? 0.0f : tsf_timecents2Secsf(p->decay));

	if (p->sustain < 0.0f) p->sustain = 0.0f;
	else if (sustainIsGain) p->sustain = tsf_decibelsToGain(-p->sustain / 10.0f);
	else p->sustain = 1.0f - (p->sustain / 1000.0f);
}

static void tsf_region_operator(struct tsf_region* region, tsf_u16 genOper, union tsf_hydra_genamount* amount)
{
	enum
	{
		StartAddrsOffset, EndAddrsOffset, StartloopAddrsOffset, EndloopAddrsOffset, StartAddrsCoarseOffset, ModLfoToPitch, VibLfoToPitch, ModEnvToPitch,
		InitialFilterFc, InitialFilterQ, ModLfoToFilterFc, ModEnvToFilterFc, EndAddrsCoarseOffset, ModLfoToVolume, Unused1, ChorusEffectsSend,
		ReverbEffectsSend, Pan, Unused2, Unused3, Unused4, DelayModLFO, FreqModLFO, DelayVibLFO, FreqVibLFO, DelayModEnv, AttackModEnv, HoldModEnv,
		DecayModEnv, SustainModEnv, ReleaseModEnv, KeynumToModEnvHold, KeynumToModEnvDecay, DelayVolEnv, AttackVolEnv, HoldVolEnv, DecayVolEnv,
		SustainVolEnv, ReleaseVolEnv, KeynumToVolEnvHold, KeynumToVolEnvDecay, Instrument, Reserved1, KeyRange, VelRange, StartloopAddrsCoarseOffset,
		Keynum, Velocity, InitialAttenuation, Reserved2, EndloopAddrsCoarseOffset, CoarseTune, FineTune, SampleID, SampleModes, Reserved3, ScaleTuning,
		ExclusiveClass, OverridingRootKey, Unused5, EndOper
	};
	switch (genOper)
	{
	case StartAddrsOffset:           region->offset += amount->shortAmount; break;
	case EndAddrsOffset:             region->end += amount->shortAmount; break;
	case StartloopAddrsOffset:       region->loop_start += amount->shortAmount; break;
	case EndloopAddrsOffset:         region->loop_end += amount->shortAmount; break;
	case StartAddrsCoarseOffset:     region->offset += amount->shortAmount * 32768; break;
	case ModLfoToPitch:              region->modLfoToPitch = amount->shortAmount; break;
	case VibLfoToPitch:              region->vibLfoToPitch = amount->shortAmount; break;
	case ModEnvToPitch:              region->modEnvToPitch = amount->shortAmount; break;
	case InitialFilterFc:            region->initialFilterFc = amount->shortAmount; break;
	case InitialFilterQ:             region->initialFilterQ = amount->shortAmount; break;
	case ModLfoToFilterFc:           region->modLfoToFilterFc = amount->shortAmount; break;
	case ModEnvToFilterFc:           region->modEnvToFilterFc = amount->shortAmount; break;
	case EndAddrsCoarseOffset:       region->end += amount->shortAmount * 32768; break;
	case ModLfoToVolume:             region->modLfoToVolume = amount->shortAmount; break;
	case Pan:                        region->pan = amount->shortAmount / 1000.0f; break;
	case DelayModLFO:                region->delayModLFO = amount->shortAmount; break;
	case FreqModLFO:                 region->freqModLFO = amount->shortAmount; break;
	case DelayVibLFO:                region->delayVibLFO = amount->shortAmount; break;
	case FreqVibLFO:                 region->freqVibLFO = amount->shortAmount; break;
	case DelayModEnv:                region->modenv.delay = amount->shortAmount; break;
	case AttackModEnv:               region->modenv.attack = amount->shortAmount; break;
	case HoldModEnv:                 region->modenv.hold = amount->shortAmount; break;
	case DecayModEnv:                region->modenv.decay = amount->shortAmount; break;
	case SustainModEnv:              region->modenv.sustain = amount->shortAmount; break;
	case ReleaseModEnv:              region->modenv.release = amount->shortAmount; break;
	case KeynumToModEnvHold:         region->modenv.keynumToHold = amount->shortAmount; break;
	case KeynumToModEnvDecay:        region->modenv.keynumToDecay = amount->shortAmount; break;
	case DelayVolEnv:                region->ampenv.delay = amount->shortAmount; break;
	case AttackVolEnv:               region->ampenv.attack = amount->shortAmount; break;
	case HoldVolEnv:                 region->ampenv.hold = amount->shortAmount; break;
	case DecayVolEnv:                region->ampenv.decay = amount->shortAmount; break;
	case SustainVolEnv:              region->ampenv.sustain = amount->shortAmount; break;
	case ReleaseVolEnv:              region->ampenv.release = amount->shortAmount; break;
	case KeynumToVolEnvHold:         region->ampenv.keynumToHold = amount->shortAmount; break;
	case KeynumToVolEnvDecay:        region->ampenv.keynumToDecay = amount->shortAmount; break;
	case KeyRange:                   region->lokey = amount->range.lo; region->hikey = amount->range.hi; break;
	case VelRange:                   region->lovel = amount->range.lo; region->hivel = amount->range.hi; break;
	case StartloopAddrsCoarseOffset: region->loop_start += amount->shortAmount * 32768; break;
	case InitialAttenuation:         region->attenuation += amount->shortAmount * 0.1f; break;
	case EndloopAddrsCoarseOffset:   region->loop_end += amount->shortAmount * 32768; break;
	case CoarseTune:                 region->transpose += amount->shortAmount; break;
	case FineTune:                   region->tune += amount->shortAmount; break;
	case SampleModes:                region->loop_mode = ((amount->wordAmount & 3) == 3 ? TSF_LOOPMODE_SUSTAIN : ((amount->wordAmount & 3) == 1 ? TSF_LOOPMODE_CONTINUOUS : TSF_LOOPMODE_NONE)); break;
	case ScaleTuning:                region->pitch_keytrack = amount->shortAmount; break;
	case ExclusiveClass:             region->group = amount->wordAmount; break;
	case OverridingRootKey:          region->pitch_keycenter = amount->shortAmount; break;
		//case gen_endOper: break; // Ignore.
		//default: addUnsupportedOpcode(generator_name);
	}
}


void LoadPresets(SF2& sf2, Presets& presets)
{
	Hydra* hydra = &sf2.hydra;
	unsigned fontSampleCount = (unsigned) sf2.fontSamples->size();
	unsigned presetNum = (unsigned)sf2.hydra.phdrs.size() - 1;
	presets.resize(presetNum);

	enum { GenInstrument = 41, GenKeyRange = 43, GenVelRange = 44, GenSampleID = 53 };

	// Read each preset.
	std::vector<tsf_hydra_phdr>::iterator pphdr, pphdrMax;
	for (pphdr = hydra->phdrs.begin(), pphdrMax = pphdr + presetNum; pphdr != pphdrMax; pphdr++)
	{
		int sortedIndex = 0, region_index = 0;
		std::vector<tsf_hydra_phdr>::iterator otherphdr;
		tsf_preset* preset;
		std::vector<tsf_hydra_pbag>::iterator ppbag, ppbagEnd;
		tsf_region globalRegion;
		for (otherphdr = hydra->phdrs.begin(); otherphdr != pphdrMax; otherphdr++)
		{
			if (otherphdr == pphdr || otherphdr->bank > pphdr->bank) continue;
			else if (otherphdr->bank < pphdr->bank) sortedIndex++;
			else if (otherphdr->preset > pphdr->preset) continue;
			else if (otherphdr->preset < pphdr->preset) sortedIndex++;
			else if (otherphdr < pphdr) sortedIndex++;
		}

		preset = &presets[sortedIndex];
		TSF_MEMCPY(preset->presetName, pphdr->presetName, sizeof(preset->presetName));
		preset->presetName[sizeof(preset->presetName) - 1] = '\0'; //should be zero terminated in source file but make sure
		preset->bank = pphdr->bank;
		preset->preset = pphdr->preset;

		unsigned regionNum = 0;

		//count regions covered by this preset
		for (ppbag = hydra->pbags.begin() + pphdr->presetBagNdx, ppbagEnd = hydra->pbags.begin() + pphdr[1].presetBagNdx; ppbag != ppbagEnd; ppbag++)
		{
			unsigned char plokey = 0, phikey = 127, plovel = 0, phivel = 127;
			std::vector<tsf_hydra_pgen>::iterator ppgen, ppgenEnd; 
			std::vector<tsf_hydra_inst>::iterator pinst; 
			std::vector<tsf_hydra_ibag>::iterator pibag, pibagEnd; 
			std::vector<tsf_hydra_igen>::iterator pigen, pigenEnd;
			for (ppgen = hydra->pgens.begin() + ppbag->genNdx, ppgenEnd = hydra->pgens.begin() + ppbag[1].genNdx; ppgen != ppgenEnd; ppgen++)
			{
				if (ppgen->genOper == GenKeyRange) { plokey = ppgen->genAmount.range.lo; phikey = ppgen->genAmount.range.hi; continue; }
				if (ppgen->genOper == GenVelRange) { plovel = ppgen->genAmount.range.lo; phivel = ppgen->genAmount.range.hi; continue; }
				if (ppgen->genOper != GenInstrument) continue;
				if (ppgen->genAmount.wordAmount >= hydra->insts.size()) continue;
				pinst = hydra->insts.begin() + ppgen->genAmount.wordAmount;
				for (pibag = hydra->ibags.begin() + pinst->instBagNdx, pibagEnd = hydra->ibags.begin() + pinst[1].instBagNdx; pibag != pibagEnd; pibag++)
				{
					unsigned char ilokey = 0, ihikey = 127, ilovel = 0, ihivel = 127;
					for (pigen = hydra->igens.begin() + pibag->instGenNdx, pigenEnd = hydra->igens.begin() + pibag[1].instGenNdx; pigen != pigenEnd; pigen++)
					{
						if (pigen->genOper == GenKeyRange) { ilokey = pigen->genAmount.range.lo; ihikey = pigen->genAmount.range.hi; continue; }
						if (pigen->genOper == GenVelRange) { ilovel = pigen->genAmount.range.lo; ihivel = pigen->genAmount.range.hi; continue; }
						if (pigen->genOper == GenSampleID && ihikey >= plokey && ilokey <= phikey && ihivel >= plovel && ilovel <= phivel) regionNum++;
					}
				}
			}
		}

		preset->regions.resize(regionNum);
		tsf_region_clear(&globalRegion, TSF_TRUE);

		// Zones.
		for (ppbag = hydra->pbags.begin() + pphdr->presetBagNdx, ppbagEnd = hydra->pbags.begin() + pphdr[1].presetBagNdx; ppbag != ppbagEnd; ppbag++)
		{
			std::vector<tsf_hydra_pgen>::iterator ppgen, ppgenEnd;
			std::vector<tsf_hydra_inst>::iterator pinst;
			std::vector<tsf_hydra_ibag>::iterator pibag, pibagEnd;
			std::vector<tsf_hydra_igen>::iterator pigen, pigenEnd;
			struct tsf_region presetRegion = globalRegion;
			int hadGenInstrument = 0;

			// Generators.
			for (ppgen = hydra->pgens.begin() + ppbag->genNdx, ppgenEnd = hydra->pgens.begin() + ppbag[1].genNdx; ppgen != ppgenEnd; ppgen++)
			{
				// Instrument.
				if (ppgen->genOper == GenInstrument)
				{
					struct tsf_region instRegion;
					tsf_u16 whichInst = ppgen->genAmount.wordAmount;
					if (whichInst >= hydra->insts.size()) continue;

					tsf_region_clear(&instRegion, TSF_FALSE);
					pinst = hydra->insts.begin()+whichInst;
					for (pibag = hydra->ibags.begin() + pinst->instBagNdx, pibagEnd = hydra->ibags.begin() + pinst[1].instBagNdx; pibag != pibagEnd; pibag++)
					{
						// Generators.
						struct tsf_region zoneRegion = instRegion;
						int hadSampleID = 0;
						for (pigen = hydra->igens.begin() + pibag->instGenNdx, pigenEnd = hydra->igens.begin() + pibag[1].instGenNdx; pigen != pigenEnd; pigen++)
						{
							if (pigen->genOper == GenSampleID)
							{
								struct tsf_hydra_shdr* pshdr;

								//preset region key and vel ranges are a filter for the zone regions
								if (zoneRegion.hikey < presetRegion.lokey || zoneRegion.lokey > presetRegion.hikey) continue;
								if (zoneRegion.hivel < presetRegion.lovel || zoneRegion.lovel > presetRegion.hivel) continue;
								if (presetRegion.lokey > zoneRegion.lokey) zoneRegion.lokey = presetRegion.lokey;
								if (presetRegion.hikey < zoneRegion.hikey) zoneRegion.hikey = presetRegion.hikey;
								if (presetRegion.lovel > zoneRegion.lovel) zoneRegion.lovel = presetRegion.lovel;
								if (presetRegion.hivel < zoneRegion.hivel) zoneRegion.hivel = presetRegion.hivel;

								//sum regions
								zoneRegion.offset += presetRegion.offset;
								zoneRegion.end += presetRegion.end;
								zoneRegion.loop_start += presetRegion.loop_start;
								zoneRegion.loop_end += presetRegion.loop_end;
								zoneRegion.transpose += presetRegion.transpose;
								zoneRegion.tune += presetRegion.tune;
								zoneRegion.pitch_keytrack += presetRegion.pitch_keytrack;
								zoneRegion.attenuation += presetRegion.attenuation;
								zoneRegion.pan += presetRegion.pan;
								zoneRegion.ampenv.delay += presetRegion.ampenv.delay;
								zoneRegion.ampenv.attack += presetRegion.ampenv.attack;
								zoneRegion.ampenv.hold += presetRegion.ampenv.hold;
								zoneRegion.ampenv.decay += presetRegion.ampenv.decay;
								zoneRegion.ampenv.sustain += presetRegion.ampenv.sustain;
								zoneRegion.ampenv.release += presetRegion.ampenv.release;
								zoneRegion.modenv.delay += presetRegion.modenv.delay;
								zoneRegion.modenv.attack += presetRegion.modenv.attack;
								zoneRegion.modenv.hold += presetRegion.modenv.hold;
								zoneRegion.modenv.decay += presetRegion.modenv.decay;
								zoneRegion.modenv.sustain += presetRegion.modenv.sustain;
								zoneRegion.modenv.release += presetRegion.modenv.release;
								zoneRegion.initialFilterQ += presetRegion.initialFilterQ;
								zoneRegion.initialFilterFc += presetRegion.initialFilterFc;
								zoneRegion.modEnvToPitch += presetRegion.modEnvToPitch;
								zoneRegion.modEnvToFilterFc += presetRegion.modEnvToFilterFc;
								zoneRegion.delayModLFO += presetRegion.delayModLFO;
								zoneRegion.freqModLFO += presetRegion.freqModLFO;
								zoneRegion.modLfoToPitch += presetRegion.modLfoToPitch;
								zoneRegion.modLfoToFilterFc += presetRegion.modLfoToFilterFc;
								zoneRegion.modLfoToVolume += presetRegion.modLfoToVolume;
								zoneRegion.delayVibLFO += presetRegion.delayVibLFO;
								zoneRegion.freqVibLFO += presetRegion.freqVibLFO;
								zoneRegion.vibLfoToPitch += presetRegion.vibLfoToPitch;

								// EG times need to be converted from timecents to seconds.
								tsf_region_envtosecs(&zoneRegion.ampenv, TSF_TRUE);
								tsf_region_envtosecs(&zoneRegion.modenv, TSF_FALSE);

								// LFO times need to be converted from timecents to seconds.
								zoneRegion.delayModLFO = (zoneRegion.delayModLFO < -11950.0f ? 0.0f : tsf_timecents2Secsf(zoneRegion.delayModLFO));
								zoneRegion.delayVibLFO = (zoneRegion.delayVibLFO < -11950.0f ? 0.0f : tsf_timecents2Secsf(zoneRegion.delayVibLFO));

								// Pin values to their ranges.
								if (zoneRegion.pan < -0.5f) zoneRegion.pan = -0.5f;
								else if (zoneRegion.pan > 0.5f) zoneRegion.pan = 0.5f;
								if (zoneRegion.initialFilterQ < 1500 || zoneRegion.initialFilterQ > 13500) zoneRegion.initialFilterQ = 0;

								pshdr = &hydra->shdrs[pigen->genAmount.wordAmount];
								zoneRegion.offset += pshdr->start;
								zoneRegion.end += pshdr->end;
								zoneRegion.loop_start += pshdr->startLoop;
								zoneRegion.loop_end += pshdr->endLoop;
								if (pshdr->endLoop > 0) zoneRegion.loop_end -= 1;
								if (zoneRegion.pitch_keycenter == -1) zoneRegion.pitch_keycenter = pshdr->originalPitch;
								zoneRegion.tune += pshdr->pitchCorrection;
								zoneRegion.sample_rate = pshdr->sampleRate;
								if (zoneRegion.end && zoneRegion.end < fontSampleCount) zoneRegion.end++;
								else zoneRegion.end = fontSampleCount;

								preset->regions[region_index] = zoneRegion;
								region_index++;
								hadSampleID = 1;
							}
							else tsf_region_operator(&zoneRegion, pigen->genOper, &pigen->genAmount);
						}

						// Handle instrument's global zone.
						if (pibag == hydra->ibags.begin() + pinst->instBagNdx && !hadSampleID)
							instRegion = zoneRegion;

						// Modulators (TODO)
						//if (ibag->instModNdx < ibag[1].instModNdx) addUnsupportedOpcode("any modulator");
					}
					hadGenInstrument = 1;
				}
				else tsf_region_operator(&presetRegion, ppgen->genOper, &ppgen->genAmount);
			}

			// Modulators (TODO)
			//if (pbag->modNdx < pbag[1].modNdx) addUnsupportedOpcode("any modulator");

			// Handle preset's global zone.
			if (ppbag == hydra->pbags.begin() + pphdr->presetBagNdx && !hadGenInstrument)
				globalRegion = presetRegion;
		}
	}
}
