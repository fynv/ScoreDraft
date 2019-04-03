#include "Python.h"

#include "Instrument.h"
#include "TrackBuffer.h"

#include "SF2.h"
#include "Presets.h"
#include "SF2Synth.h"
#include <math.h>

class SF2Instrument : public Instrument
{
public:
	SF2Instrument()
	{
		m_input = nullptr;
		m_preset = nullptr;

	}
	virtual ~SF2Instrument() {} 

	void setSF2(F32Samples* input, tsf_preset* preset) { m_input = input; m_preset = preset; }

	virtual bool IsGMDrum()
	{
		if (!m_preset) return false;
		return m_preset->bank == 128;
	}

protected:
	virtual void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
	{
		if (!m_input || !m_preset) return;

		float freq = sampleFreq*noteBuf->m_sampleRate;
		float key = (float)(log((double)freq / 261.626) / log(2.0)*12.0 + 60.0);

		unsigned numSamples = (unsigned)(fNumOfSamples + 0.5f);

		F32Samples_deferred f32Samples = SF2Synth(*m_input, *m_preset, key, 1.0f, numSamples, STEREO_INTERLEAVED, noteBuf->m_sampleRate);

		noteBuf->m_sampleNum = (unsigned)f32Samples->size()/2;
		noteBuf->m_channelNum = 2;
		noteBuf->Allocate();

		memcpy(noteBuf->m_data, f32Samples->data(), sizeof(float)*f32Samples->size());
	}

private:
	F32Samples* m_input;
	tsf_preset* m_preset;

};

class SF2InstrumentInitializer
{
public:
	std::string m_sf2Path;

	SF2InstrumentInitializer()
	{
		m_initialized = false;
	}

	SF2Instrument* Init(unsigned preset_index)
	{
		if (!m_initialized) _initialize();

		SF2Instrument* inst = new SF2Instrument();
		inst->setSF2(m_fontSamples, &(*m_presets)[preset_index]);

		return inst;
	}

	void ListPresets()
	{
		if (!m_initialized) _initialize();
		for (unsigned i = 0; i < m_presets->size(); i++)
		{
			tsf_preset& preset = (*m_presets)[i];
			printf("%d : %s bank=%u number=%u\n", i, preset.presetName, preset.bank, preset.preset);
		}
	}

private:
	void _initialize()
	{
		Deferred<SF2> sf2 = LoadSF2Filename(m_sf2Path.data());
		m_fontSamples = sf2->fontSamples;
		m_presets = LoadPresets(sf2);
		m_initialized = true;
	}
	bool m_initialized;
	F32Samples_deferred m_fontSamples;
	Presets_deferred m_presets;
};

#include <map>
std::map<std::string, SF2InstrumentInitializer> s_initializers;

SF2InstrumentInitializer* GetInitializer(std::string path)
{
	if (s_initializers.find(path) == s_initializers.end())
	{
		SF2InstrumentInitializer initializer;
		initializer.m_sf2Path = path;
		s_initializers[path] = initializer;
	}
	return &s_initializers[path];
}

static PyObject* SF2InstrumentListPresets(PyObject *self, PyObject *args)
{
	std::string path = _PyUnicode_AsString(PyTuple_GetItem(args, 0));
	SF2InstrumentInitializer* initializer = GetInitializer(path);
	initializer->ListPresets();
	return PyLong_FromUnsignedLong(0);
}

static PyObject* InitializeSF2Instrument(PyObject *self, PyObject *args)
{
	std::string path = _PyUnicode_AsString(PyTuple_GetItem(args, 0));
	unsigned preset_index = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	SF2InstrumentInitializer* initializer = GetInitializer(path);
	SF2Instrument* inst = initializer->Init(preset_index);
	return PyLong_FromVoidPtr(inst);
}


static PyObject* DestroySF2Instrument(PyObject *self, PyObject *args)
{
	SF2Instrument* inst = (SF2Instrument*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	delete inst;
	return PyLong_FromLong(0);
}


static PyMethodDef s_Methods[] = {
	{
		"SF2InstrumentListPresets",
		SF2InstrumentListPresets,
		METH_VARARGS,
		""
	},
	{
		"InitializeSF2Instrument",
		InitializeSF2Instrument,
		METH_VARARGS,
		""
	},
	{
		"DestroySF2Instrument",
		DestroySF2Instrument,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"SF2Instrument_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	s_Methods
};

PyMODINIT_FUNC PyInit_PySF2Instrument(void) {
	return PyModule_Create(&cModPyDem);
}


