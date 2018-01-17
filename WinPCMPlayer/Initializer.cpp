#include <Python.h>
#include "PyScoreDraft.h"
#include "WinPCMPlayer.h"

static PyScoreDraft* s_pPyScoreDraft;
static WinPCMPlayer s_Player;

PyObject * PlayTrackBuffer(PyObject *args)
{
	unsigned BufferId = (unsigned)PyLong_AsUnsignedLong(args);
	TrackBuffer_deferred buffer = s_pPyScoreDraft->GetTrackBuffer(BufferId);
	s_Player.PlayTrack(*buffer);
	return PyLong_FromUnsignedLong(0);
}

PyObject * PlayGetRemainingTime(PyObject *args)
{
	return PyFloat_FromDouble( (double)s_Player.GetRemainingTime());
}

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	s_pPyScoreDraft = pyScoreDraft;
	
	pyScoreDraft->RegisterInterfaceExtension("PlayTrackBuffer", PlayTrackBuffer, "buf", "buf.id",
		"\t'''\n"
		"\tUsing Win32 API to playback a track-buffer.\n"
		"\tbuf -- an instance of TrackBuffer.\n"
		"\tNote that this function is a async call. Please keep the main thread busy or do a sleep to let the playback continue.\n"
		"\t'''\n");

	pyScoreDraft->RegisterInterfaceExtension("PlayGetRemainingTime", PlayGetRemainingTime, "", "",
		"\t'''\n"
		"\tMonitoring how much time in seconds is remaining in current play-back.\n"
		"\t'''\n");

}
