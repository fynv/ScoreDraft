#include <Python.h>
#include "WinPCMPlayer.h"

static WinPCMPlayer s_Player;

static PyObject* PlayTrackBuffer(PyObject *self, PyObject *args)
{
	TrackBuffer* buffer = (TrackBuffer*)PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
	s_Player.PlayTrack(*buffer);
	return PyLong_FromUnsignedLong(0);
}

static PyObject* PlayGetRemainingTime(PyObject *self, PyObject *args)
{
	return PyFloat_FromDouble( (double)s_Player.GetRemainingTime());
}


static PyMethodDef s_Methods[] = {
	{
		"PlayTrackBuffer",
		PlayTrackBuffer,
		METH_VARARGS,
		""
	},
	{
		"PlayGetRemainingTime",
		PlayGetRemainingTime,
		METH_VARARGS,
		""
	},
	{ NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem =
{
	PyModuleDef_HEAD_INIT,
	"WinPCMPlayer_module", /* name of module */
	"",          /* module documentation, may be NULL */
	-1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
	s_Methods
};

PyMODINIT_FUNC PyInit_PyWinPCMPlayer(void) {
	return PyModule_Create(&cModPyDem);
}

