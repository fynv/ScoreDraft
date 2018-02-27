#ifndef YF_COMPLEX
#define YF_COMPLEX
typedef struct
{
	double Re;
	double Im;
}DComp;
double DCEnergy(const DComp* c);
double DCAbs(const DComp*);
double DCAng(const DComp*);
void DCSetAA(DComp*,double,double);
void DCConjugate(DComp*,const DComp*);
void DCAdd(DComp*,const DComp*,const DComp*);
void DCSub(DComp*,const DComp*,const DComp*);
void DCMul(DComp*,const DComp*,const DComp*);
void DCCW90(DComp*,const DComp*);
void DCCCW90(DComp*,const DComp*);
void DCPowN(DComp*,const DComp*,int n);
#endif