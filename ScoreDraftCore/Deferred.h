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
		if (m != nullptr) m->addRef();
	}
	~Deferred()
	{
		if (m != nullptr) m->release();
		m = nullptr;
	}

	void Abondon()
	{
		if (m != nullptr) m->release();
		m = nullptr;
	}

	void operator=(const Deferred & in)
	{
		if (in.m != nullptr) in.m->addRef();
		if (m != nullptr) m->release();
		m = in.m;
	}

	T* operator -> () 
	{ 
		if (m == nullptr) return nullptr;
		return m->t; 
	}
	const T* operator -> () const 
	{ 
		if (m == nullptr) return nullptr;
		return m->t; 
	}

	operator T*() 
	{ 
		if (m == nullptr) return nullptr;
		return m->t; 
	}

	operator const T*() const 
	{ 
		if (m == nullptr) return nullptr;
		return m->t; 
	}

	template <class SubClass>
	static Deferred<T> Instance()
	{
		SubClass* dummy=0;
		return Deferred<T>(dummy);
	}

	template <class SubClass>
	SubClass* DownCast()
	{
		if (m == nullptr) return nullptr;
		return (SubClass*)m->t;
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

