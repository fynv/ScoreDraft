#include <vector>

template <class T>
struct VectorView
{
	unsigned count;
	T* d_data;
};

template <class T>
class DVVector
{
public:
	typedef VectorView<T> ViewType;
	DVVector()
	{
		m_count = 0;
		m_data = nullptr;
	}

	virtual ~DVVector()
	{
		Free();
	}

	ViewType view()
	{
		return{ m_count, m_data };
	}

	unsigned Count() const
	{
		return m_count;
	}

	T* Pointer()
	{
		return m_data;
	}

	const T* ConstPointer() const
	{
		return m_data;
	}

	operator T*()
	{
		return m_data;
	}

	operator const T*()
	{
		return m_data;
	}

	void Free()
	{
		if (m_data != nullptr)
		{
			cudaFree(m_data);
			m_data = nullptr;
		}
		m_count = 0;
	}

	void Allocate(unsigned count)
	{
		Free();
		m_count = count;
		if (m_count>0)
			cudaMalloc(&m_data, sizeof(T)*count);
	}

	const DVVector& operator = (const std::vector<T>& cpuVec)
	{
		Free();
		Allocate((unsigned)cpuVec.size());
		if (m_count > 0)
		{
			cudaMemcpy(m_data, cpuVec.data(), sizeof(T)*m_count, cudaMemcpyHostToDevice);
		}
		return *this;
	}

	void ToCPU(std::vector<T>& cpuVec) const
	{
		cpuVec.resize(m_count);
		cudaMemcpy(cpuVec.data(), m_data, sizeof(T)*m_count, cudaMemcpyDeviceToHost);
	}

	void Update(const std::vector<T>& cpuVec)
	{
		if (cpuVec.size() != m_count)
			*this = cpuVec;
		else if (m_count > 0)
			cudaMemcpy(m_data, cpuVec.data(), sizeof(T)*m_count, cudaMemcpyHostToDevice);
	}

protected:
	unsigned m_count;
	T* m_data;
};

template <class T_GPU, class T_CPU>
class DVImagedVector : public DVVector<typename T_GPU::ViewType>
{
public:
	void Allocate(const std::vector<unsigned>& counts)
	{
		unsigned count = (unsigned)counts.size();
		m_vecs.resize(count);
		std::vector<typename T_GPU::ViewType> tmp(count);
		for (size_t i = 0; i < count; i++)
		{
			m_vecs[i].Allocate(counts[i]);
			tmp[i] = m_vecs[i].view();
		}
		DVVector<typename T_GPU::ViewType>::operator = (tmp);
	}

	const DVImagedVector& operator = (const std::vector<T_CPU>& cpuVecs)
	{
		unsigned count = (unsigned)cpuVecs.size();
		m_vecs.resize(count);
		std::vector<typename T_GPU::ViewType> tmp(count);
		for (unsigned i = 0; i < count; i++)
		{
			m_vecs[i] = cpuVecs[i];
			tmp[i] = m_vecs[i].view();
		}
		DVVector<typename T_GPU::ViewType>::operator = (tmp);
		return *this;
	}

	void ToCPU(std::vector<T_CPU>& cpuVecs) const
	{
		unsigned count = (unsigned)m_vecs.size();
		cpuVecs.resize(count);
		for (unsigned i = 0; i < count; i++)
			m_vecs[i].ToCPU(cpuVecs[i]);
	}

	void Update(const std::vector<T_CPU>& cpuVecs)
	{
		if (cpuVecs.size() != m_vecs.size())
			*this = cpuVecs;
		else
		{
			size_t count = m_vecs.size();
			std::vector<typename T_GPU::ViewType> tmp(count);
			for (size_t i = 0; i < count; i++)
			{
				m_vecs[i].Update(cpuVecs[i]);
				tmp[i] = m_vecs[i].view();
			}
			DVVector<typename T_GPU::ViewType>::Update(tmp);
		}
	}

private:
	std::vector<T_GPU> m_vecs;
};

template <class T>
using DVLevel2Vector = DVImagedVector <DVVector<T>, std::vector<T>>;
