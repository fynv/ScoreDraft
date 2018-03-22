#ifdef HAVE_CUDA
#include "UtauDraft.h"

void UtauDraft::GenWaveStruct::_generateWave_CUDA_HNM()
{
	printf("here!\n");
	_generateWave_HNM();
}

#endif