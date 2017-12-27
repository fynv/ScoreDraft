#ifndef _PyScoreDraft_h
#define _PyScoreDraft_h

#include <string>
#include <vector>
#include "Instrument.h"
#include "Percussion.h"

typedef Deferred<Instrument> Instrument_deferred;
typedef Deferred<Percussion> Percussion_deferred;

class InstrumentFactory
{
public:
	virtual void GetInstrumentList(std::vector<std::string>& list){}	
	virtual void InitiateInstrument(unsigned clsInd, Instrument_deferred& inst) {}

	virtual void GetPercussionList(std::vector<std::string>& list){}
	virtual void InitiatePercussion(unsigned clsInd, Percussion_deferred& perc) {}
};

typedef Instrument_deferred(InstrumentInitializer)();
typedef Percussion_deferred(PercussionInitializer)();

template <class T_Instrument>
Instrument_deferred t_InstInitializer()
{
	return Instrument_deferred::Instance<T_Instrument>();
}

template <class T_Percussion>
Percussion_deferred t_PercInitializer()
{
	return Percussion_deferred::Instance<T_Percussion>();
}

class TypicalInstrumentFactory : public InstrumentFactory
{
public:
	template <class T_Instrument>
	void AddInstrument(const char* name)
	{
		m_InstrumentList.push_back(name);
		m_InstrumentInitializers.push_back(t_InstInitializer<T_Instrument>);
	}

	template <class T_Percussion>
	void AddPercussion(const char* name)
	{
		m_PercussionList.push_back(name);
		m_PercussionInitializers.push_back(t_PercInitializer<T_Percussion>);
	}

	virtual void GetInstrumentList(std::vector<std::string>& list)
	{
		list = m_InstrumentList;
	}

	virtual void InitiateInstrument(unsigned clsInd, Instrument_deferred& inst)
	{
		inst = m_InstrumentInitializers[clsInd]();
	}

	virtual void GetPercussionList(std::vector<std::string>& list)
	{
		list = m_PercussionList;
	}

	virtual void InitiatePercussion(unsigned clsInd, Percussion_deferred& perc)
	{
		perc = m_PercussionInitializers[clsInd]();
	}

protected:
	std::vector<std::string> m_InstrumentList;
	std::vector<InstrumentInitializer*> m_InstrumentInitializers;

	std::vector<std::string> m_PercussionList;
	std::vector<PercussionInitializer*> m_PercussionInitializers;
};

#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C" __declspec(dllexport) InstrumentFactory*


#endif

