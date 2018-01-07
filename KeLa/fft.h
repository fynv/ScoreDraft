#ifndef YF_FFT
#define YF_FFT
#include "complex.h"
#define PI 3.1415926535897932384626433832795
void fft(DComp *a,unsigned l);
void ifft(DComp *a,unsigned l);
#endif