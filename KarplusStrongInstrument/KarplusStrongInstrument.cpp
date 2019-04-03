#include "Python.h"
#include "fft.h"
#include <vector>
#include "Instrument.h"
#include "TrackBuffer.h"

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}

static void GeneratePinkNoise(float period, std::vector<float>& ret)
{
	unsigned uLen = (unsigned)ceilf(period);
	unsigned l = 0;
	unsigned fftLen = 1;
	while (fftLen < uLen)
	{
		fftLen <<= 1;
		l++;
	}

	std::vector<DComp> fftData(fftLen);
	memset(&fftData[0], 0, sizeof(DComp)*fftLen);

	for (unsigned i = 1; i < (unsigned)(period) / 2; i++)
	{
		float amplitude = (float)fftLen / sqrtf((float)i);
		float phase = rand01()*(float)(2.0*PI);
		fftData[i].Re = (double)(amplitude*cosf(phase));
		fftData[i].Im = (double)(amplitude*sinf(phase));

		fftData[fftLen - i].Re = fftData[i].Re;
		fftData[fftLen - i].Im = -fftData[i].Im;
	}

	ifft(&fftData[0], l);

	unsigned pnLen = (unsigned)ceilf(period*2.0f);

	ret.resize(pnLen);

	float rate = (float)fftLen / period;
	for (unsigned i = 0; i < pnLen; i++)
	{
		int ipos1 = (int)ceilf(((float)i - 0.5f)*rate);
		if (ipos1 < 0) ipos1 = 0;
		int ipos2 = (int)floorf(((float)i + 0.5f)*rate);
		int count = ipos2 - ipos1 + 1;

		float sum = 0.0f;
		for (int ipos = ipos1; ipos <= ipos2; ipos++)
		{
			int _ipos = ipos;
			while (_ipos >= fftLen) _ipos -= fftLen;
			sum += (float)fftData[_ipos].Re;
		}
		ret[i] = sum / (float)count;
	}
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
		std::vector<float> pinkNoise;
		GeneratePinkNoise(period, pinkNoise);
		
		float sustainLen = m_sustain_periods*period;
		unsigned totalLen = (unsigned)ceilf(fNumOfSamples + sustainLen);

		noteBuf->m_sampleNum = totalLen;
		noteBuf->m_channelNum = 1;
		noteBuf->Allocate();

		float cut_freq = m_cut_freq/261.626f* sampleFreq;
		float a = (float)(1.0 - exp(-2.0*PI* cut_freq));

		unsigned pos = 0;

		while (pos < totalLen)
		{
			float value = 0.0f;
			if ((float)pos < period*2.0f)
				value += pinkNoise[pos] * 0.5f*(cosf(((float)pos - period) / period*PI) + 1.0f);

			if ((float)pos >= period)
			{
				float gain = (float)pos < fNumOfSamples ? m_loop_gain : m_sustain_gain;

				float refPos = (float)pos - period;

				int refPos1 = (int)refPos;
				int refPos2 = refPos1 + 1;
				float frac = refPos - (float)refPos1;

				// linear interpolation
				float ref = noteBuf->m_data[refPos1] * (1.0f - frac) + noteBuf->m_data[refPos2] * frac;

				// cubic interpolation
				/*int refPos0 = refPos1 - 1;
				if (refPos0 < 0) refPos0 = 0;

				int refPos3 = refPos1 + 2;
				float p0 = noteBuf->m_data[refPos0];
				float p1 = noteBuf->m_data[refPos1];
				float p2 = noteBuf->m_data[refPos2];
				float p3 = noteBuf->m_data[refPos3];
				float ref = (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*powf(frac, 3.0f) +
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0 + 0.5f*p2)*frac + p1;*/

				value += gain*a*ref + (1.0f - a)*noteBuf->m_data[pos - 1];
			}
			
			noteBuf->m_data[pos] = value;
			pos++;
		}

	}

private:
	float m_sustain_gain;
	float m_sustain_periods;

	float m_loop_gain;
	float m_cut_freq;
};

static PyObject* InitializeKarplusStrongInstrument(PyObject *self, PyObject *args)
{
	KarplusStrongInstrument* inst = new KarplusStrongInstrument;
	return PyLong_FromVoidPtr(inst);
}


static PyObject* DestroyKarplusStrongInstrument(PyObject *self, PyObject *args)
{
	KarplusStrongInstrument* inst = (KarplusStrongInstrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete inst;
	return PyLong_FromLong(0);
}

static PyObject* KarplusStrongSetCutFrequency(PyObject *self, PyObject *args)
{
	KarplusStrongInstrument* inst = (KarplusStrongInstrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float cut_freq = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	inst->SetCutFrequency(cut_freq);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* KarplusStrongSetLoopGain(PyObject *self, PyObject *args)
{
	KarplusStrongInstrument* inst = (KarplusStrongInstrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float loop_gain = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	inst->SetLoopGain(loop_gain);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* KarplusStrongSetSustainGain(PyObject *self, PyObject *args)
{
	KarplusStrongInstrument* inst = (KarplusStrongInstrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	float sustain_gain = (float)PyFloat_AsDouble(PyTuple_GetItem(args, 1));
	inst->SetSustainGain(sustain_gain);
	return PyLong_FromUnsignedLong(0);
}


static PyMethodDef s_Methods[] = {
	{
		"InitializeKarplusStrongInstrument",
		InitializeKarplusStrongInstrument,
		METH_VARARGS,
		""
	},
	{
		"DestroyKarplusStrongInstrument",
		DestroyKarplusStrongInstrument,
		METH_VARARGS,
		""
	},
	{
		"KarplusStrongSetCutFrequency",
		KarplusStrongSetCutFrequency,
		METH_VARARGS,
		""
	},
	{
		"KarplusStrongSetLoopGain",
		KarplusStrongSetLoopGain,
		METH_VARARGS,
		""
	},
	{
		"KarplusStrongSetSustainGain",
		KarplusStrongSetSustainGain,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"KarplusStrong_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	s_Methods
};

PyMODINIT_FUNC PyInit_PyKarplusStrongInstrument(void) {
	return PyModule_Create(&cModPyDem);
}


