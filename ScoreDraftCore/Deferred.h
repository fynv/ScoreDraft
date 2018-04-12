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
	virtual ~Deferred()
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
		return Deferred<T>(new SubClass);
	}

	template <class SubClass>
	SubClass* DownCast()
	{
		if (m == nullptr) return nullptr;
		return (SubClass*)m->t;
	}

protected:
	Deferred(T* t) : m(new Deferred<T>::Internal(t))
	{
		m->addRef();
	}

private:
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

