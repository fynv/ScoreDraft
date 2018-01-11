#ifndef _PyScoreDraft_h
#define _PyScoreDraft_h

#include <string>
#include <vector>
#include <utility>
#include <Deferred.h>

#include "Instrument.h"
#include "Percussion.h"
#include "Singer.h"
#include "TrackBuffer.h"

typedef Deferred<Instrument> Instrument_deferred;
typedef Deferred<Percussion> Percussion_deferred;
typedef Deferred<Singer> Singer_deferred;

class InstrumentInitializer
{
public:
	virtual Instrument_deferred Init() = 0;
};

class PercussionInitializer
{
public:
	virtual Percussion_deferred Init() = 0;
};

class SingerInitializer
{
public:
	virtual Singer_deferred Init() = 0;
};

template <class T_Instrument>
class t_InstInitializer : public InstrumentInitializer
{
public:
	virtual Instrument_deferred Init()
	{
		return Instrument_deferred::Instance<T_Instrument>();
	}
};

template <class T_Percussion>
class t_PercInitializer : public PercussionInitializer
{
public:
	virtual Percussion_deferred Init()
	{
		return Percussion_deferred::Instance<T_Percussion>();
	}
};

template <class T_Singer>
class t_SingerInitializer : public SingerInitializer
{
public:
	virtual Singer_deferred Init()
	{
		return Singer_deferred::Instance<T_Singer>();
	}
};

typedef std::pair<std::string, InstrumentInitializer*> InstrumentClass;
typedef std::vector<InstrumentClass> InstrumentClassList;
typedef std::pair<std::string, PercussionInitializer*> PercussionClass;
typedef std::vector<PercussionClass> PercussionClassList;
typedef std::pair<std::string, SingerInitializer*> SingerClass;
typedef std::vector<SingerClass> SingerClassList;

typedef std::vector<Instrument_deferred> InstrumentMap;
typedef std::vector<Percussion_deferred> PercussionMap;
typedef std::vector<Singer_deferred> SingerMap;
typedef std::vector<TrackBuffer_deferred> TrackBufferMap;

class Logger
{
public:
	virtual void PrintLine(const char* line) const = 0;
};

class PyScoreDraft
{
public:
	PyScoreDraft() { m_logger = nullptr; }
	void SetLogger(const Logger* logger)
	{
		m_logger = logger;
	}

	void RegisterInstrumentClass(const char* name, InstrumentInitializer* initializer)
	{
		if (m_logger != nullptr)
		{
			char line[1024];
			sprintf(line, "Registering instrument, clsId=%lu, name=%s", m_InstrumentClasses.size(), name);
			m_logger->PrintLine(line);
		}
		m_InstrumentClasses.push_back(InstrumentClass(name, initializer));
		
	}
	void RegisterPercussionClass(const char* name, PercussionInitializer* initializer)
	{
		if (m_logger != nullptr)
		{
			char line[1024];
			sprintf(line, "Registering Percussion, clsId=%lu, name=%s", m_PercussionClasses.size(), name);
			m_logger->PrintLine(line);
		}
		m_PercussionClasses.push_back(PercussionClass(name, initializer));
	}
	void RegisterSingerClass(const char* name, SingerInitializer* initializer)
	{
		if (m_logger != nullptr)
		{
			char line[1024];
			sprintf(line, "Registering Singer, clsId=%lu, name=%s", m_SingerClasses.size(), name);
			m_logger->PrintLine(line);
		}
		m_SingerClasses.push_back(SingerClass(name, initializer));
	}

	unsigned NumOfIntrumentClasses()
	{
		return (unsigned)m_InstrumentClasses.size();
	}

	InstrumentClass GetInstrumentClass(unsigned i)
	{
		return m_InstrumentClasses[i];
	}

	unsigned NumOfPercussionClasses()
	{
		return (unsigned)m_PercussionClasses.size();
	}

	PercussionClass GetPercussionClass(unsigned i)
	{
		return m_PercussionClasses[i];
	}

	unsigned NumOfSingerClasses()
	{
		return (unsigned)m_SingerClasses.size();
	}

	SingerClass GetSingerClass(unsigned i)
	{
		return m_SingerClasses[i];
	}

	unsigned NumOfInstruments()
	{
		return (unsigned)m_InstrumentMap.size();
	}

	unsigned NumOfPercussions()
	{
		return (unsigned)m_PercussionMap.size();
	}

	unsigned NumOfSingers()
	{
		return (unsigned)m_SingerMap.size();
	}

	unsigned AddInstrument(Instrument_deferred instrument)
	{
		unsigned id = (unsigned)m_InstrumentMap.size();
		m_InstrumentMap.push_back(instrument);
		return id;
	}

	unsigned AddPercussion(Percussion_deferred perc)
	{
		unsigned id = (unsigned)m_PercussionMap.size();
		m_PercussionMap.push_back(perc);
		return id;
	}

	unsigned AddSinger(Singer_deferred singer)
	{
		unsigned id = (unsigned)m_SingerMap.size();
		m_SingerMap.push_back(singer);
		return id;
	}

	Instrument_deferred GetInstrument(unsigned id)
	{
		return m_InstrumentMap[id];
	}

	Percussion_deferred GetPercussion(unsigned id)
	{
		return m_PercussionMap[id];
	}

	Singer_deferred GetSinger(unsigned id)
	{
		return m_SingerMap[id];
	}

	unsigned NumOfTrackBuffers()
	{
		return (unsigned)m_TrackBufferMap.size();
	}

	unsigned AddTrackBuffer(TrackBuffer_deferred buffer)
	{
		unsigned id = (unsigned)m_TrackBufferMap.size();
		m_TrackBufferMap.push_back(buffer);
		return id;
	}

	TrackBuffer_deferred GetTrackBuffer(unsigned id)
	{
		return m_TrackBufferMap[id];
	}

private:
	const Logger* m_logger;

	InstrumentClassList m_InstrumentClasses;
	PercussionClassList m_PercussionClasses;
	SingerClassList m_SingerClasses;

	TrackBufferMap m_TrackBufferMap;

	InstrumentMap m_InstrumentMap;
	PercussionMap m_PercussionMap;
	SingerMap m_SingerMap;
	
};


#ifdef _WIN32
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C" __declspec(dllexport)
#else
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C"
#endif

#endif

