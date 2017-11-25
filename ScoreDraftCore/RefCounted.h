#ifndef _scoredraft_RefCounted_h
#define _scoredraft_RefCounted_h

class RefCounted
{
public:
	RefCounted() : m_count(0)
	{
	}

	virtual ~RefCounted()
	{
	}

	unsigned addRef() const
	{
		m_count++;
		return m_count;
	}

	unsigned release() const
	{
		m_count--;
		if (m_count == 0) {
			delete this;
			return 0;
		}
		return m_count;
	}

	int refCount() const
	{
		return m_count;
	}

private: 
	mutable int m_count;

	RefCounted(const RefCounted &); 
	RefCounted &operator=(const RefCounted &);
};

#endif
