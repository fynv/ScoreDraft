#include "PyScoreDraft.h"

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

	Instrument_deferred Init(unsigned preset_index)
	{
		if (!m_initialized) _initialize();

		Instrument_deferred inst = Instrument_deferred::Instance<SF2Instrument>();
		inst.DownCast<SF2Instrument>()->setSF2(m_fontSamples, &(*m_presets)[preset_index]);

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

static PyScoreDraft* s_PyScoreDraft;

PyObject * SF2InstrumentListPresets(PyObject *args)
{
	std::string path = _PyUnicode_AsString(args);
	SF2InstrumentInitializer* initializer = GetInitializer(path);
	initializer->ListPresets();
	return PyLong_FromUnsignedLong(0);
}

PyObject * InitializeSF2Instrument(PyObject *args)
{
	std::string path = _PyUnicode_AsString(PyTuple_GetItem(args, 0));
	unsigned preset_index = (unsigned)PyLong_AsUnsignedLong(PyTuple_GetItem(args, 1));
	SF2InstrumentInitializer* initializer = GetInitializer(path);
	Instrument_deferred inst = initializer->Init(preset_index);
	unsigned id = s_PyScoreDraft->AddInstrument(inst);
	return PyLong_FromUnsignedLong(id);
}


PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft, const char* root)
{
	s_PyScoreDraft = pyScoreDraft;
	pyScoreDraft->RegisterInterfaceExtension("InitializeSF2Instrument", InitializeSF2Instrument,
		"sf2Path, preset_index", "sf2Path, preset_index",
		"\t'''\n"
		"\tInitialize a SF2 based instrument\n"
		"\tsf2Path -- path to the sf2 file.\n"
		"\tpreset_index -- preset index.\n"
		"\t'''\n");
	pyScoreDraft->RegisterInterfaceExtension("SF2InstrumentListPresets", SF2InstrumentListPresets,
		"sf2Path", "sf2Path",
		"\t'''\n"
		"\tList presets of a sf2 file\n"
		"\tsf2Path -- path to the sf2 file.\n"
		"\t'''\n");
}



