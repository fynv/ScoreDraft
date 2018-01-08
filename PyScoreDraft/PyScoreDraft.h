#ifndef _PyScoreDraft_h
#define _PyScoreDraft_h

#include <string>
#include <vector>
#include "Instrument.h"
#include "Percussion.h"
#include "Singer.h"

typedef Deferred<Instrument> Instrument_deferred;
typedef Deferred<Percussion> Percussion_deferred;
typedef Deferred<Singer> Singer_deferred;

class InstrumentFactory
{
public:
	virtual void GetInstrumentList(std::vector<std::string>& list){}	
	virtual void InitiateInstrument(unsigned clsInd, Instrument_deferred& inst) {}

	virtual void GetPercussionList(std::vector<std::string>& list){}
	virtual void InitiatePercussion(unsigned clsInd, Percussion_deferred& perc) {}

	virtual void GetSingerList(std::vector<std::string>& list){}
	virtual void InitiateSinger(unsigned clsInd, Singer_deferred& singer) {}
};

typedef Instrument_deferred(InstrumentInitializer)();
typedef Percussion_deferred(PercussionInitializer)();
typedef Singer_deferred(SingerInitializer)();

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


template <class T_Singer>
Singer_deferred t_InstInitializer()
{
	return Singer_deferred::Instance<T_Singer>();
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

	template <class T_Singer>
	void AddSinger(const char* name)
	{
		m_SingerList.push_back(name);
		m_SingerInitializers.push_back(t_InstInitializer<T_Singer>);
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

	virtual void GetSingerList(std::vector<std::string>& list)
	{
		list = m_SingerList;
	}

	virtual void InitiateSinger(unsigned clsInd, Singer_deferred& singer)
	{
		singer = m_SingerInitializers[clsInd]();
	}

protected:
	std::vector<std::string> m_InstrumentList;
	std::vector<InstrumentInitializer*> m_InstrumentInitializers;

	std::vector<std::string> m_PercussionList;
	std::vector<PercussionInitializer*> m_PercussionInitializers;

	std::vector<std::string> m_SingerList;
	std::vector<SingerInitializer*> m_SingerInitializers;
};

#ifdef _WIN32
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C" __declspec(dllexport) InstrumentFactory*
#else
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C" InstrumentFactory*
#endif

#endif

