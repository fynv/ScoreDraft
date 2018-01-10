#include "fft.h"
#include "math.h"
void fft(DComp *a,unsigned l)
{
	DComp u,w,t;	
	unsigned n=1,nv2,i,j,k;
	unsigned le,lei,ip,m;
	double tmp;
	n<<=l;
	nv2=n>>1;
	j=0;
	for (i=0;i<n-1;i++)
	{
		if (i<j)
		{
			t=a[j];
			a[j]=a[i];
			a[i]=t;
		}
		k=nv2;
		while (k<=j)
		{
			j-=k;
			k>>=1;
		}
		j+=k;
	}
	le=1;
	for(m=1;m<=l;m++)
	{
		lei=le;
		le<<=1;
		u.Re=1;	u.Im=0;
		tmp=PI/lei;
		w.Re=cos(tmp); w.Im=-sin(tmp);
		for (j=0;j<lei;j++)
		{
			for (i=j;i<n;i+=le)
			{
				ip=i+lei;
				DCMul(&t,&u,a+ip);
				DCSub(a+ip,a+i,&t);
				DCAdd(a+i,a+i,&t);
			}
			DCMul(&u,&u,&w);
		}
	}
}
void ifft(DComp *a,unsigned l)
{
	DComp u,w,t;	
	unsigned n=1,nv2,i,j,k;
	unsigned le,lei,ip,m;
	double tmp;
	n<<=l;
	nv2=n>>1;
	j=0;
	for (i=0;i<n-1;i++)
	{
		if (i<j)
		{
			t=a[j];
			a[j]=a[i];
			a[i]=t;
		}
		k=nv2;
		while (k<=j)
		{
			j-=k;
			k>>=1;
		}
		j+=k;
	}
	le=1;
	for(m=1;m<=l;m++)
	{
		lei=le;
		le<<=1;
		u.Re=0.5;	u.Im=0;
		tmp=PI/lei;
		w.Re=cos(tmp); w.Im=sin(tmp);
		for (j=0;j<lei;j++)
		{
			for (i=j;i<n;i+=le)
			{
				ip=i+lei;
				DCMul(&t,&u,a+ip);
				(a+i)->Im*=0.5;
				(a+i)->Re*=0.5;
				DCSub(a+ip,a+i,&t);
				DCAdd(a+i,a+i,&t);
			}
			DCMul(&u,&u,&w);
		}
	}
}