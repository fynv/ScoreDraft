#include "Synth.h"
#include <cmath>

void Synth(const float* input, float* outputBuffer, unsigned numSamples, NoteState& noteState, const SynthCtrl& control)
{
	float* outL = outputBuffer;
	float* outR = (control.outputmode == STEREO_UNWEAVED ? outL + numSamples : nullptr);

	unsigned tmpLoopStart = control.loopStart;
	unsigned tmpLoopEnd = control.loopEnd;
	unsigned tmpEnd = control.end;
	double tmpSourceSamplePosition = noteState.sourceSamplePosition;

	double tmpSampleEndDbl = (double)tmpEnd;
	double tmpLoopEndDbl = (double)tmpLoopEnd + 1.0;

	unsigned i_ctrl = 0;
	SynthCtrlPnt ctrlPnt;

	LowPassState lowPassState = noteState.lowPass;

	while (numSamples)
	{
		float gainLeft, gainRight;
		int blockSamples = (numSamples > control.effect_sample_block ? control.effect_sample_block : numSamples);
		numSamples -= blockSamples;

		if (i_ctrl<control.controlPnts.size())
			ctrlPnt = control.controlPnts[i_ctrl];
		else
			break;

		float gainMono = ctrlPnt.gainMono;
		double pitchRatio = ctrlPnt.pitchRatio;
		bool interpolation = pitchRatio <= 1.0f;

		gainLeft = gainMono *control.panFactorLeft;
		gainRight = gainMono  * control.panFactorRight;

		LowPassCtrlPnt lowPassCtrlPnt = ctrlPnt.lowPass;

		while (blockSamples-- && tmpSourceSamplePosition < tmpSampleEndDbl)
		{
			float val = 0.0f;
			if (interpolation)
			{
				int ipos1 = (int)tmpSourceSamplePosition;
				float frac = (float)(tmpSourceSamplePosition - (double)ipos1);
				int ipos2 = ipos1 + 1;
				int ipos3 = ipos1 + 2;
				int ipos0 = ipos1 - 1;

				if (ipos1 > (int)tmpLoopEnd && ctrlPnt.looping)
				{
					ipos2 = tmpLoopStart;
					ipos3 = tmpLoopStart + 1;
				}
				if (ipos2 >= (int)tmpEnd) ipos2 = tmpEnd - 1;
				if (ipos3 >= (int)tmpEnd) ipos3 = tmpEnd - 1;
				if (ipos0 < 0) ipos0 = 0;

				float p0 = input[ipos0];
				float p1 = input[ipos1];
				float p2 = input[ipos2];
				float p3 = input[ipos3];

				val = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;
			}
			else
			{
				int ipos1 = (int)ceil(tmpSourceSamplePosition - 0.5* pitchRatio);
				int ipos2 = (int)floor(tmpSourceSamplePosition + 0.5* pitchRatio);
				int count = ipos2 - ipos1 + 1;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					int _ipos = ipos;
					if (_ipos < 0) _ipos = 0;
					if (_ipos > (int)tmpLoopEnd && ctrlPnt.looping)
					{
						_ipos += (int)tmpLoopStart - (int)tmpLoopEnd -1;
					}
					if (_ipos >= (int)tmpEnd)
					{
						_ipos = tmpEnd - 1;
					}
					val += input[_ipos];
				}
				val /= (float)count;
			}

			if (lowPassCtrlPnt.active)
			{
				double In = val;
				val = (float)(In * lowPassCtrlPnt.a0 + lowPassState.z1);
				lowPassState.z1 = In * lowPassCtrlPnt.a1 + lowPassState.z2 - lowPassCtrlPnt.b1 * val; 
				lowPassState.z2 = In * lowPassCtrlPnt.a0 - lowPassCtrlPnt.b2 * val; 
			}

			switch (control.outputmode)
			{
			case STEREO_INTERLEAVED:
				*outL++ += val * gainLeft;
				*outL++ += val * gainRight;
				break;
			case STEREO_UNWEAVED:
				*outL++ += val * gainLeft;
				*outR++ += val * gainRight;
				break;
			case MONO:
				*outL++ += val * gainMono;
				break;
			}
			// Next sample.
			tmpSourceSamplePosition += pitchRatio;
			if (tmpSourceSamplePosition >= tmpLoopEndDbl && ctrlPnt.looping) 
				tmpSourceSamplePosition -= (tmpLoopEnd - tmpLoopStart + 1.0f);

		}

		if (tmpSourceSamplePosition >= tmpSampleEndDbl)
			break;

		i_ctrl++;
	}

	noteState.sourceSamplePosition= tmpSourceSamplePosition;
	noteState.lowPass = lowPassState;
}
