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

	void Abondon()
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

	T* operator -> () 
	{ 
		if (m == NULL) return NULL;
		return m->t; 
	}
	const T* operator -> () const 
	{ 
		if (m == NULL) return NULL;
		return m->t; 
	}

	operator T*() 
	{ 
		if (m == NULL) return NULL;
		return m->t; 
	}

	operator const T*() const 
	{ 
		if (m == NULL) return NULL;
		return m->t; 
	}

	template <class SubClass>
	static Deferred<T> Instance()
	{
		SubClass* dummy=0;
		return Deferred<T>(dummy);
	}

private:
	template <class SubClass>
	Deferred(SubClass* t) : m(new Deferred<T>::Internal(t))
	{
		m->addRef();
	}
	class Internal : public RefCounted 
	{
	public:
		T* t;
		Internal()
		{
			t = new T;
		}
	
		template <class SubClass>
		Internal(SubClass* dummy)
		{
			t = new SubClass;
		}

		~Internal()
		{
			delete t;
		}
	};
	Internal* m;
};

#endif

