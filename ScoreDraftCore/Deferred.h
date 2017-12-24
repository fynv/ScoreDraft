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
	Deferred(T* t) : m(new Deferred<T>::Internal(t))
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

	T* operator -> () { return m->t; }
	const T* operator -> () const { return m->t; }

	operator T*() { return m->t; }
	operator const T*() const { return m->t; }

	class Internal : public RefCounted 
	{
	public:
		T* t;
		Internal()
		{
			t = new T;
		}
		Internal(T* t)
		{
			this->t = t;
		}
		~Internal()
		{
			delete t;
		}
	};
	Internal* m;
};

#endif

