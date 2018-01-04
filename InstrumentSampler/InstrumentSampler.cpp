#include "PyScoreDraft.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <dlfcn.h>
#endif

#include <string.h>
#include <math.h>

#include "fft.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


struct WavHeader
{
	unsigned short wFormatTag;
	unsigned short wChannels;
	unsigned int dwSamplesPerSec;
	unsigned int dwAvgBytesPerSec;
	unsigned short wBlockAlign;
	unsigned short wBitsPerSample;
};

class InstrumentSampler : public Instrument
{
public:
	InstrumentSampler()
	{
		m_wav_length = 0;
		m_wav_samples = 0;
	}

	~InstrumentSampler()
	{
		delete[] m_wav_samples;
	}

	bool LoadWav(const char* name)
	{
		char filename[1024];
		sprintf(filename, "InstrumentSamples/%s.wav", name);

		delete[] m_wav_samples;
		m_wav_length = 0;
		m_wav_samples = 0;

		char c_riff[4] = { 'R', 'I', 'F', 'F' };
		unsigned& u_riff = *(unsigned*)c_riff;

		char c_wave[4] = { 'W', 'A', 'V', 'E' };
		unsigned& u_wave = *(unsigned*)c_wave;

		char c_fmt[4] = { 'f', 'm', 't', ' ' };
		unsigned& u_fmt = *(unsigned*)c_fmt;

		char c_data[4] = { 'd', 'a', 't', 'a' };
		unsigned& u_data = *(unsigned*)c_data;

		unsigned buf32;

		FILE *fp = fopen(filename, "rb");
		if (!fp) return false;
		fread(&buf32, 4, 1, fp);
		if (buf32 != u_riff)
		{
			fclose(fp);
			return false;
		}

		unsigned chunkSize;
		fread(&chunkSize, 4, 1, fp);

		fread(&buf32, 4, 1, fp);
		if (buf32 != u_wave)
		{
			fclose(fp);
			return false;
		}

		fread(&buf32, 4, 1, fp);
		if (buf32 != u_fmt)
		{
			fclose(fp);
			return false;
		}

		unsigned headerSize;
		fread(&headerSize, 4, 1, fp);
		if (headerSize != sizeof(WavHeader))
		{
			fclose(fp);
			return false;
		}

		WavHeader header;
		fread(&header, sizeof(WavHeader), 1, fp);

		if (header.wFormatTag != 1)
		{
			fclose(fp);
			return false;
		}

		unsigned channels = header.wChannels;
		if (channels<1 || channels>2)
		{
			fclose(fp);
			return false;
		}

		m_origin_sample_rate = header.dwSamplesPerSec;

		if (header.wBitsPerSample != 16)
		{
			fclose(fp);
			return false;
		}

		fread(&buf32, 4, 1, fp);
		if (buf32 != u_data)
		{
			fclose(fp);
			return false;
		}

		unsigned dataSize;
		fread(&dataSize, 4, 1, fp);

		short* data = new short[dataSize / 2];

		fread(data, sizeof(short), dataSize / 2, fp);

		m_wav_length = dataSize / channels / 2;
		m_wav_samples = new float[m_wav_length];

		m_max_v = 0.0f;

		for (unsigned i = 0; i < m_wav_length; i++)
		{
			float v = 0.0f;
			for (unsigned j = 0; j < channels; j++)
			{
				v += (float)data[i * channels + j];
			}
			v /= 32767.0f*(float)channels;
			m_wav_samples[i] = v;
			m_max_v = max(m_max_v, fabsf(v));

		}

		delete[] data;

		fclose(fp);

		_fetchOriginFreq(name);

		return true;
	}

private:
	void _fetchOriginFreq(const char* name)
	{
		char filename[1024];
		sprintf(filename, "InstrumentSamples/%s.freq", name);

		FILE *fp = fopen(filename, "r");
		if (fp)
		{
			fscanf(fp, "%f", &m_origin_freq);
			fclose(fp);
		}
		else
		{
			unsigned len = 1;
			unsigned l = -1;
			while (len <= m_wav_length)
			{
				l++;
				len <<= 1;
			}
			len = 1 << l;

			DComp* fftData = new DComp[len];

			for (unsigned i = 0; i<len; i++)
			{
				fftData[i].Re = (double)m_wav_samples[i];
				fftData[i].Im = 0.0;
			}
			fft(fftData, l);

			// self-correlation
			for (unsigned i = 0; i<len; i++)
			{
				DComp c = fftData[i];
				fftData[i].Re = c.Re*c.Re+ c.Im*c.Im;
				fftData[i].Im = 0.0;
			}

			ifft(fftData, l);

			double thresh = fftData[0].Re*0.7;
			double lastV = fftData[0].Re;
			bool ascending = false;
			unsigned maxi = 0;
			for (unsigned i = 1; i < min(m_origin_sample_rate / 30, len); i++)
			{
				double v = fftData[i].Re;
				if (v > thresh)
				{
					if (!ascending)
					{
						if (v > lastV) ascending = true;
					}
					else
					{
						if (v < lastV)
						{
							maxi = i - 1;
							break;
						}
					}
					lastV = v;
				}
			}

			m_origin_freq = (float)m_origin_sample_rate / (float)maxi;


			/*
			// FFT Peak
			unsigned starti = 100 * len /  (float)m_origin_sample_rate;
			unsigned stopi = 2000 * len /  (float)m_origin_sample_rate;
			unsigned maxi = starti;
			double max = DCAbs(fftData + maxi);
			for (unsigned i = starti + 1; i < stopi; i++)
			{
				if (max<DCAbs(fftData + i))
				{
					max = DCAbs(fftData + i);
					maxi = i;
				}
			}
			delete[] fftData;

			m_origin_freq=(float)maxi* (float)m_origin_sample_rate / (float)len;*/

			printf("Detected frequency of %s.wav = %fHz\n", name, m_origin_freq);
			fp = fopen(filename, "w");
			fprintf(fp,"%f\n", m_origin_freq);
			fclose(fp);
		}
	}
	
	void GenerateNoteWave(float fNumOfSamples, float sampleFreq, NoteBuffer* noteBuf)
	{
		float origin_SampleFreq = m_origin_freq / (float)m_origin_sample_rate;
		unsigned maxSample = (unsigned)((float)m_wav_length*origin_SampleFreq / sampleFreq);

		noteBuf->m_sampleNum = min((unsigned)ceilf(fNumOfSamples), maxSample);
		noteBuf->Allocate();

		float mult = m_noteVolume / m_max_v;

		bool interpolation = sampleFreq <= origin_SampleFreq;

		for (unsigned j = 0; j < noteBuf->m_sampleNum; j++)
		{
			float x2 = (float)j / fNumOfSamples;
			float amplitude = 1.0f - expf((x2 - 1.0f)*10.0f);

			float wave;
			if (interpolation)
			{
				float pos = (float)j *sampleFreq / origin_SampleFreq;
				int ipos1 = (int)pos;
				float frac = pos - (float)ipos1;
				int ipos2 = ipos1 + 1;
				if (ipos2 >= (int)m_wav_length) ipos2 = (int)m_wav_length-1;

				// linear interpolation
				//wave = m_wav_samples[ipos1] * (1.0f - frac) + m_wav_samples[ipos2] * frac;
				
				// cubic interpolation
				int ipos0 = ipos1 - 1;
				if (ipos0 < 0) ipos0 = 0;

				int ipos3 = ipos1 + 2;
				if (ipos3 >= (int)m_wav_length) ipos3 = (int)m_wav_length-1;

				float p0 = m_wav_samples[ipos0];
				float p1 = m_wav_samples[ipos1];
				float p2 = m_wav_samples[ipos2];
				float p3 = m_wav_samples[ipos3];

				wave = (-0.5f*p0+1.5f*p1-1.5f*p2+0.5f*p3)*powf(frac, 3.0f)+
					(p0 - 2.5f*p1 + 2.0f*p2 - 0.5f*p3)*powf(frac, 2.0f) +
					(-0.5f*p0+0.5f*p2)*frac+p1;
			}
			else
			{
				int ipos1 = (int)ceilf(((float)j - 0.5f)*sampleFreq / origin_SampleFreq);
				int ipos2 = (int)floorf(((float)j + 0.5f)*sampleFreq / origin_SampleFreq);
				if (ipos1 < 0) ipos1 = 0;
				if (ipos2 >= (int)m_wav_length) ipos2 = (int)m_wav_length - 1;
				int count = ipos2 - ipos1 + 1;
				float sum = 0.0f;
				for (int ipos = ipos1; ipos <= ipos2; ipos++)
				{
					sum += m_wav_samples[ipos];
				}
				wave = sum / (float)count;
			}

			noteBuf->m_data[j] = amplitude*wave;
		}
	}

	unsigned m_wav_length;
	float *m_wav_samples;
	float m_max_v;
	float m_origin_freq;
	unsigned m_origin_sample_rate;

};

class InstrumentSamplerFactory : public InstrumentFactory
{
public:
	InstrumentSamplerFactory()
	{
#ifdef _WIN32
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = INVALID_HANDLE_VALUE;

		hFind = FindFirstFileA("InstrumentSamples\\*.wav", &ffd);
		if (INVALID_HANDLE_VALUE == hFind) return;

		do
		{
			if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;

			char name[1024];
			memcpy(name, ffd.cFileName, strlen(ffd.cFileName) - 4);
			name[strlen(ffd.cFileName) - 4] = 0;
			m_InstList.push_back(name);

		} while (FindNextFile(hFind, &ffd) != 0);

#else
		DIR *dir;
	    struct dirent *entry;

	    if (dir = opendir("InstrumentSamples"))
	    {
	    	while ((entry = readdir(dir)) != NULL)
	    	{
	    		const char* ext=entry->d_name+ strlen(entry->d_name)-4;
	    		if (strcmp(ext,".wav")==0)
	    		{
	    			char name[1024];
					memcpy(name, entry->d_name, strlen(entry->d_name) - 4);
					name[strlen(entry->d_name) - 4] = 0;
					m_InstList.push_back(name);
	    		}

	    	}

	    }

#endif

	}

	virtual void GetInstrumentList(std::vector<std::string>& list)
	{
		list = m_InstList;
	}

	virtual void InitiateInstrument(unsigned clsInd, Instrument_deferred& inst)
	{
		inst = Instrument_deferred::Instance<InstrumentSampler>();
		inst.DownCast<InstrumentSampler>()->LoadWav(m_InstList[clsInd].data());
	}

private:
	std::vector<std::string> m_InstList;

};

PY_SCOREDRAFT_EXTENSION_INTERFACE GetFactory()
{
	static InstrumentSamplerFactory fac;
	return &fac;
}

