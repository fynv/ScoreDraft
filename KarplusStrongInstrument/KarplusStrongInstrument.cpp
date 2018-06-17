#include "PyScoreDraft.h"
#include "fft.h"
#include <vector>

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}

Deferred<std::vector<float>> GeneratePinkNoise(unsigned uLen)
{
	unsigned l = 0;
	unsigned fftLen = 1;
	while (fftLen < uLen)
	{
		fftLen <<= 1;
		l++;
	}

	std::vector<DComp> fftData(fftLen);
	memset(&fftData[0], 0, sizeof(DComp)*fftLen);

	for (unsigned i = 1; i < uLen / 2; i++)
	{
		float amplitude = (float)fftLen / sqrtf((float)i);
		float phase = rand01()*(float)(2.0*PI);
		fftData[i].Re = (double)(amplitude*cosf(phase));
		fftData[i].Im = (double)(amplitude*sinf(phase));

		fftData[fftLen - i].Re = fftData[i].Re;
		fftData[fftLen - i].Im = -fftData[i].Im;
	}

	ifft(&fftData[0], l);

	Deferred<std::vector<float>> ret;
	ret->resize(uLen);

	float rate = (float)fftLen / (float)uLen;
	for (unsigned i = 0; i < uLen; i++)
	{
		int ipos1 = (int)ceilf(((float)i - 0.5f)*rate);
		if (ipos1 < 0) ipos1 = 0;
		int ipos2 = (int)floorf(((float)i + 0.5f)*rate);
		if (ipos2 >=(int) fftLen) ipos2 = fftLen - 1;
		int count = ipos2 - ipos1 + 1;

		float sum = 0.0f;
		for (int ipos = ipos1; ipos <= ipos2; ipos++)
		{
			sum += (float)fftData[ipos].Re;
		}
		(*ret)[i]=sum / (float)count;
	}
	return ret;
}

class KarplusStrongInstrument : public Instrument
{
public:
	KarplusStrongInstrument()
	{
		m_sustain_gain = 0.8f;
		m_sustain_periods = logf(0.01f) / logf(m_sustain_gain);
		m_loop_gain = 0.99f;
		m_cut_freq = 10000.0f;
	}

	~KarplusStrongInstrument() {}

	void SetCutFrequency(float cut_freq)
	{
		m_cut_freq = cut_freq;
	}

	void SetLoopGain(float loop_gain)
	{
		m_loop_gain = loop_gain;
	}

	void SetSustainGain(float sustain_gain)
	{
		m_sustain_gain = sustain_gain;
		m_sustain_periods = logf(0.01f) / logf(m_sustain_gain);
	}

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
	{
		float period = 1.0f / sampleFreq;
		Deferred<std::vector<float>> pinkNoise = GeneratePinkNoise((unsigned)ceilf(period));
		
		float sustainLen = m_sustain_periods*period;
		unsigned totalLen = (unsigned)ceilf(fNumOfSamples + sustainLen);

		noteBuf->m_sampleNum = totalLen;
		noteBuf->m_channelNum = 1;
		noteBuf->Allocate();

		memcpy(noteBuf->m_data, pinkNoise->data(), sizeof(float)*pinkNoise->size());

		unsigned pos = (unsigned)pinkNoise->size();

		float cut_freq = m_cut_freq/261.626f* sampleFreq;
		float a = (float)(1.0 - exp(-2.0*PI* cut_freq));

		while (pos < totalLen)
		{
			float gain = (float)pos < fNumOfSamples ? m_loop_gain : m_sustain_gain;

			float refPos = (float)pos - period;
			unsigned refPos0 = (unsigned)refPos;
			unsigned refPos1 = refPos0 + 1;
			float k = refPos - (float)refPos0;
			float ref = noteBuf->m_data[refPos0] * (1.0f - k) + noteBuf->m_data[refPos1] * k;
			
			noteBuf->m_data[pos] = gain*a*ref + (1.0f - a)*noteBuf->m_data[pos - 1];

			pos++;
		}

	}

private:
	float m_sustain_gain;
	float m_sustain_periods;

	float m_loop_gain;
	float m_cut_freq;
};

static PyScoreDraft* s_PyScoreDraft;

PyObject * InitializeKarplusStrongInstrument(PyObject *args)
{
	Instrument_deferred inst = Instrument_deferred::Instance<KarplusStrongInstrument>();
	unsigned id = s_PyScoreDraft->AddInstrument(inst);
	return PyLong_FromUnsignedLong(id);
}

PyObject* KarplusStrongSetCutFrequency(PyObject *args)
{
	unsigned InstId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	float cut_freq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));

	Instrument_deferred inst = s_PyScoreDraft->GetInstrument(InstId);
	inst.DownCast<KarplusStrongInstrument>()->SetCutFrequency(cut_freq);

	return PyLong_FromUnsignedLong(0);
}

PyObject* KarplusStrongSetLoopGain(PyObject *args)
{
	unsigned InstId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	float loop_gain = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));

	Instrument_deferred inst = s_PyScoreDraft->GetInstrument(InstId);
	inst.DownCast<KarplusStrongInstrument>()->SetLoopGain(loop_gain);

	return PyLong_FromUnsignedLong(0);
}

PyObject* KarplusStrongSetSustainGain(PyObject *args)
{
	unsigned InstId = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 0));
	float sustain_gain = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));

	Instrument_deferred inst = s_PyScoreDraft->GetInstrument(InstId);
	inst.DownCast<KarplusStrongInstrument>()->SetSustainGain(sustain_gain);

	return PyLong_FromUnsignedLong(0);
}



PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	s_PyScoreDraft = pyScoreDraft;
	pyScoreDraft->RegisterInterfaceExtension("InitializeKarplusStrongInstrument", InitializeKarplusStrongInstrument,
		"", "",
		"\t'''\n"
		"\tInitialize a KarplusStrongInstrument.\n"
		"\t'''\n");

	pyScoreDraft->RegisterInterfaceExtension("KarplusStrongSetCutFrequency", KarplusStrongSetCutFrequency, "inst, cut_freq", "inst.id, cut_freq");
	pyScoreDraft->RegisterInterfaceExtension("KarplusStrongSetLoopGain", KarplusStrongSetLoopGain, "inst, loop_gain", "inst.id, loop_gain");
	pyScoreDraft->RegisterInterfaceExtension("KarplusStrongSetSustainGain", KarplusStrongSetSustainGain, "inst, sustain_gain", "inst.id, sustain_gain");

}

