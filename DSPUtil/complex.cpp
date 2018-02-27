#include <math.h>
#include "stdio.h"
#include "complex.h"

double DCEnergy(const DComp* c)
{
	return c->Im*c->Im + c->Re*c->Re;
}

double DCAbs(const DComp* c)
{
	return sqrt(c->Im*c->Im+c->Re*c->Re);
}
double DCAng(const DComp* c)
{
	return atan2(c->Im,c->Re);
}
void DCSetAA(DComp* c,double Abs,double Ang)
{
	c->Re=Abs*cos(Ang);
	c->Im=Abs*sin(Ang);
}
void DCConjugate(DComp* c,const DComp* c1)
{
	c->Re=c1->Re;
	c->Im=-c1->Im;
}

void DCAdd(DComp* c,const DComp* c1,const DComp* c2)
{
	c->Re=c1->Re+c2->Re;
	c->Im=c1->Im+c2->Im;
}
void DCSub(DComp* c,const DComp* c1,const DComp* c2)
{
	c->Re=c1->Re-c2->Re;
	c->Im=c1->Im-c2->Im;
}
void DCMul(DComp* c,const DComp* c1,const DComp* c2)
{
	DComp temp;
	temp.Re=c1->Re*c2->Re-c1->Im*c2->Im;
	temp.Im=c1->Re*c2->Im+c1->Im*c2->Re;
	*c=temp;
}
void DCCW90(DComp* c,const DComp* c1)
{
	DComp temp;
	temp.Im=c1->Re;
	temp.Re=-c1->Im;
	*c=temp;
}
void DCCCW90(DComp* c,const DComp* c1)
{
	DComp temp;
	temp.Im=-c1->Re;
	temp.Re=c1->Im;
	*c=temp;
}
void DCPowN(DComp* c,const DComp* c1,int n)
{
	DCSetAA(c,pow(DCAbs(c1),n),DCAng(c1)*n);
}
void DCPrint(const DComp *c,FILE* f)
{
	if (c->Im==0) fprintf(f,"%g",c->Re);
	if (c->Im>0) fprintf(f,"%g+%gj",c->Re,c->Im);
	if (c->Im<0) fprintf(f,"%g%gj",c->Re,c->Im);
}