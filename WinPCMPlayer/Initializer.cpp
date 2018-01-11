#include <Python.h>
#include "PyScoreDraft.h"
#include "stdlib.h"
#include "WinPCMPlayer.h"

#include "Deferred.h"

static PyScoreDraft* s_pPyScoreDraft;
static Deferred<WinPCMPlayer> s_pPlayer;

PyObject * PlayTrackBuffer(PyObject *args)
{
	unsigned BufferId = (unsigned)PyLong_AsUnsignedLong(args);
	TrackBuffer_deferred buffer = s_pPyScoreDraft->GetTrackBuffer(BufferId);
	s_pPlayer->PlayTrack(*buffer);
	return PyLong_FromUnsignedLong(0);
}

PY_SCOREDRAFT_EXTENSION_INTERFACE void Initialize(PyScoreDraft* pyScoreDraft)
{
	s_pPyScoreDraft = pyScoreDraft;
	pyScoreDraft->RegisterInterfaceExtension("PlayTrackBuffer", PlayTrackBuffer,"buf", "", "buf.id");

}
