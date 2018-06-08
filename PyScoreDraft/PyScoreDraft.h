#ifndef _PyScoreDraft_h
#define _PyScoreDraft_h

#include <Python.h>

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

typedef std::vector<Instrument_deferred> InstrumentMap;
typedef std::vector<Percussion_deferred> PercussionMap;
typedef std::vector<Singer_deferred> SingerMap;
typedef std::vector<TrackBuffer_deferred> TrackBufferMap;

typedef PyObject *(*PyScoreDraftExtensonFunc)(PyObject *param);

struct InterfaceExtension
{
	std::string m_name;
	PyScoreDraftExtensonFunc m_func;
	std::string m_input_params;
	std::string m_call_params;
	std::string m_comment;
};

typedef std::vector<InterfaceExtension> InterfaceExtensionList;

class Logger
{
public:
	virtual ~Logger(){}
	virtual void PrintLine(const char* line) const = 0;
};

class PyScoreDraft
{
public:
	PyScoreDraft();
	void SetLogger(const Logger* logger)
	{
		m_logger = logger;
	}

	void RegisterInterfaceExtension(const char* name, PyScoreDraftExtensonFunc func,
		const char* input_params="", const char* call_params="", const char* comment = "")
	{
		if (m_logger != nullptr)
		{
			char line[1024];
			sprintf(line, "Registering Extension, extId=%lu, name=%s", (unsigned)m_InterfaceExtensions.size(), name);
			m_logger->PrintLine(line);
		}

		InterfaceExtension ext;
		ext.m_name = name; 
		ext.m_func = func;
		ext.m_input_params = input_params;
		ext.m_call_params = call_params;
		ext.m_comment = comment;

		m_InterfaceExtensions.push_back(ext);
	}


	unsigned NumOfInterfaceExtensions()
	{
		return (unsigned)m_InterfaceExtensions.size();
	}

	InterfaceExtension GetInterfaceExtension(unsigned i)
	{
		return m_InterfaceExtensions[i];
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

	PyMethodDef* GetPyScoreDraftMethods()
	{
		return m_PyScoreDraftMethods;
	}

private:
	const Logger* m_logger;

	InterfaceExtensionList m_InterfaceExtensions;

	TrackBufferMap m_TrackBufferMap;

	InstrumentMap m_InstrumentMap;
	PercussionMap m_PercussionMap;
	SingerMap m_SingerMap;

	PyMethodDef* m_PyScoreDraftMethods;
	
};


#ifdef _WIN32
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C" __declspec(dllexport)
#else
#define PY_SCOREDRAFT_EXTENSION_INTERFACE extern "C"
#endif

#endif

