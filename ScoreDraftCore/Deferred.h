#ifndef _scoredraft_Deferred_h
#define _scoredraft_Deferred_h

#include "RefCounted.h"

template<class T>
class Deferred
{
public:
	Deferred() : m(new Deferred<T>::Internal())
	{
		m->addRef();
	}
	Deferred(const Deferred & in) : m(in.m)
	{
		if (m != NULL) m->addRef();
	}
	~Deferred()
	{
		if (m != NULL) m->release();
		m = NULL;
	}

	void operator=(const Deferred & in)
	{
		if (in.m != NULL) in.m->addRef();
		if (m != NULL) m->release();
		m = in.m;
	}

	T* operator -> () { return m; }
	const T* operator -> () const { return m; }

	operator T*() { return m; }
	operator const T*() const { return m; }

	class Internal : public RefCounted, public T {};
	Internal* m;
};

#endif

