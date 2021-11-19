#include <thread>
#include <mutex>
#include <vector>
#include <list>
#include <queue>
#include <memory.h>
#include <cmath>
#include <portaudio.h>
#include <GLFW/glfw3.h>

#include <TrackBuffer.h>
#include <utils.h>
#include "PCMPlayer.h"


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


class TrackBufferCopy
{
public:
	TrackBufferCopy(TrackBuffer& input)
	{
		m_AlignPos = input.AlignPos();
		m_chn = input.NumberOfChannels();
		unsigned size = input.NumberOfSamples();
		m_data.resize(size*m_chn);

		float volume = input.AbsoluteVolume();
		float pan = input.Pan();

		for (unsigned i = 0; i < size; i++)
		{
			float sample[2];
			input.Sample(i, sample);
			if (m_chn == 1)
			{
				m_data[i] = (short)(max(min(sample[0] * volume, 1.0f), -1.0f)*32767.0f);
			}
			else if (m_chn == 2)
			{
				CalcPan(pan, sample[0], sample[1]);
				m_data[2 * i] = (short)(max(min(sample[0] * volume, 1.0f), -1.0f)*32767.0f);
				m_data[2 * i + 1] = (short)(max(min(sample[1] * volume, 1.0f), -1.0f)*32767.0f);
			}
		}
	}

	~TrackBufferCopy()
	{

	}

	std::vector<short> m_data;
	unsigned m_AlignPos;
	unsigned m_chn;
};

class TrackBufferQueue
{
public:
	TrackBufferQueue()
	{

	}

	~TrackBufferQueue()
	{

	}

	size_t Size()
	{
		return m_queue.size();
	}

	void Front2(TrackBufferCopy*& track0, TrackBufferCopy*& track1)
	{
		track0 = nullptr;
		track1 = nullptr;
		m_mutex.lock();
		auto iter = m_queue.begin();
		if (iter != m_queue.end())
		{
			track0 = *iter;
			iter++;
			if (iter != m_queue.end())
			{
				track1 = *iter;
			}
		}		
		m_mutex.unlock();
	}

	void Push(TrackBufferCopy* buf)
	{
		m_mutex.lock();
		m_queue.push_back(buf);
		m_mutex.unlock();
	}

	TrackBufferCopy* Pop()
	{
		m_mutex.lock();
		TrackBufferCopy* ret = m_queue.front();
		m_queue.pop_front();
		m_mutex.unlock();
		return ret;
	}

private:
	std::list<TrackBufferCopy*> m_queue;
	std::mutex m_mutex;
};

class AudioBufferQueue
{
public:
	AudioBufferQueue(int cache_size) : m_semaphore_in(cache_size)
	{
	}

	~AudioBufferQueue()
	{
	}

	size_t Size()
	{
		return m_queue.size();
	}

	AudioBuffer* Front()
	{
		return m_queue.front();
	}

	void Push(AudioBuffer* packet)
	{
		m_semaphore_in.wait();
		m_mutex.lock();
		m_queue.push(packet);
		m_mutex.unlock();
		m_semaphore_out.notify();
	}

	AudioBuffer* Pop()
	{
		m_semaphore_out.wait();
		m_mutex.lock();
		AudioBuffer* ret = m_queue.front();
		m_queue.pop();
		m_mutex.unlock();
		m_semaphore_in.notify();
		return ret;
	}

private:
	std::queue<AudioBuffer*> m_queue;
	std::mutex m_mutex;
	Semaphore m_semaphore_in;
	Semaphore m_semaphore_out;
};


int PCMPlayer::stream_callback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	unsigned long statusFlags,
	void *userData)
{
	PCMPlayer* self = (PCMPlayer*)userData;
	AudioBufferQueue* queue = self->m_audio_queue.get();
	unsigned long pos_out = 0;
	short *out = (short*)outputBuffer;
	while (pos_out < framesPerBuffer)
	{
		if (queue->Size() < 1) break;
		AudioBuffer* buf_in = queue->Front();		
		size_t copy_size = min(AUDIO_BUF_LEN - self->m_audio_read_pos, framesPerBuffer - pos_out);
		memcpy(out + pos_out * 2, buf_in->data + self->m_audio_read_pos * 2, sizeof(short)*copy_size * 2);
		self->m_audio_read_pos += copy_size;
		pos_out += (unsigned long)copy_size;
		if (self->m_audio_read_pos == AUDIO_BUF_LEN)
		{
			self->m_audio_read_pos = 0;
			queue->Pop();
			delete buf_in;			
		}
	}
	if (pos_out < framesPerBuffer)
		memset(out + pos_out * 2, 0, sizeof(short) * (framesPerBuffer - pos_out) * 2);

	return paContinue;
}

void PCMPlayer::thread_demux(PCMPlayer* self)
{
	TrackBufferQueue* queue_in = self->m_input_queue.get();
	AudioBufferQueue* queue_out = self->m_audio_queue.get();
	while (self->m_demuxing)
	{
		AudioBuffer* buf_out = new AudioBuffer;
		size_t pos_out = 0;
		while (pos_out < AUDIO_BUF_LEN)
		{
			TrackBufferCopy *buf_in, *buf_in2;
			queue_in->Front2(buf_in, buf_in2);
			if (buf_in == nullptr) break;

			size_t size_in = buf_in->m_data.size() / buf_in->m_chn;
			size_t copy_size = min(size_in - self->m_input_read_pos, AUDIO_BUF_LEN - pos_out);
			
			for (size_t i = 0; i < copy_size; i++)
			{
				size_t read_pos = self->m_input_read_pos + i;
				short value[2] = { 0,0 };
				if (buf_in->m_chn == 1)
				{
					value[0] = value[1] = buf_in->m_data[read_pos];
				}
				else if (buf_in->m_chn == 2)
				{
					value[0] = buf_in->m_data[read_pos * 2];
					value[1] = buf_in->m_data[read_pos * 2 + 1];
				}

				if (buf_in2 != nullptr && read_pos > size_in - buf_in2->m_AlignPos)
				{
					unsigned read_pos2 = (unsigned)read_pos - (unsigned)(size_in - buf_in2->m_AlignPos);					
					short value2[2] = { 0,0 };
					if (buf_in2->m_chn == 1)
					{
						value2[0] = value2[1] = buf_in2->m_data[read_pos2];
					}
					else if (buf_in2->m_chn == 2)
					{
						value2[0] = buf_in2->m_data[read_pos2 * 2];
						value2[1] = buf_in2->m_data[read_pos2 * 2 + 1];
					}

					for (unsigned j = 0; j < 2; j++)
					{
						int v = (int)value[j] + (int)value2[j];
						if (v > 32767) v = 32767;
						else if (v < -32767) v = -32767;
						value[j] = (short)v;
					}
				}

				size_t write_pos = pos_out + i;
				buf_out->data[write_pos * 2] = value[0];
				buf_out->data[write_pos * 2 + 1] = value[1];
			}	

			pos_out += copy_size;

			self->m_mutex_progress.lock();
			self->m_input_read_pos += copy_size;
			if (self->m_input_read_pos == size_in)
			{				
				self->m_total_length -= buf_in->m_data.size() / buf_in->m_chn;
				if (buf_in2 != nullptr)
				{
					self->m_input_read_pos = buf_in2->m_AlignPos;
				}
				else
				{
					self->m_input_read_pos = 0;
				}				
				self->m_mutex_progress.unlock();

				queue_in->Pop();
				delete buf_in;
			}
			else
			{
				self->m_mutex_progress.unlock();
			}		
			
		}
		if (pos_out < AUDIO_BUF_LEN)
			memset(buf_out->data + pos_out * 2, 0, sizeof(short) * (AUDIO_BUF_LEN - pos_out) * 2);

		self->m_ring_buf[self->m_next_buf] = *buf_out;
		self->m_next_buf = (self->m_next_buf + 1) % 12;
		queue_out->Push(buf_out);		
	}
}

thread_local int PCMPlayer::s_instance_count = 0;

PCMPlayer::PCMPlayer(double sample_rate, bool ui) : m_sample_rate(sample_rate)
{
	if (s_instance_count == 0) Pa_Initialize();
	s_instance_count++;

	m_input_queue = std::unique_ptr<TrackBufferQueue>(new TrackBufferQueue);
	m_audio_queue = std::unique_ptr<AudioBufferQueue>(new AudioBufferQueue(8));

	m_demuxing = true;
	m_thread_demux = (std::unique_ptr<std::thread>)(new std::thread(thread_demux, this));

	PaStreamParameters outputParameters = {};
	outputParameters.device = Pa_GetDefaultOutputDevice();
	outputParameters.channelCount = 2;
	outputParameters.sampleFormat = paInt16;
	outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;

	Pa_OpenStream(&m_stream, nullptr, &outputParameters, sample_rate, (unsigned long)(sample_rate*0.01), paClipOff, stream_callback, this);
	Pa_StartStream(m_stream);
	
	if (ui)
	{
		glfwInit();		
		m_window = glfwCreateWindow(640, 320, "ScoreDraft PCM Player", NULL, NULL);
		glfwMakeContextCurrent(m_window);
		glfwSwapInterval(1);
		glfwSetWindowUserPointer(m_window, this);
		glfwSetKeyCallback(m_window, key_callback);
		printf("Press 'W' to show waveform.\n");
		printf("Press 'S' to show spectrum.\n");
	}
}

PCMPlayer::~PCMPlayer()
{
	if (m_window != nullptr)
	{
		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

	m_demuxing = false;

	Pa_StopStream(m_stream);
	Pa_CloseStream(m_stream);

	while (m_audio_queue->Size() > 0)
	{
		AudioBuffer* p = m_audio_queue->Pop();
		delete p;
	}

	while (m_input_queue->Size() > 0)
	{
		TrackBufferCopy* p = m_input_queue->Pop();
		delete p;
	}

	m_thread_demux->join();
	m_thread_demux = nullptr;

	s_instance_count--;
	if (s_instance_count == 0) Pa_Terminate();
}

void PCMPlayer::PlayTrack(TrackBuffer &track)
{
	m_input_queue->Push(new TrackBufferCopy(track));
	m_mutex_progress.lock();
	m_total_length += track.NumberOfSamples();
	m_mutex_progress.unlock();
}

float PCMPlayer::GetRemainingTime()
{
	m_mutex_progress.lock();
	float remaining = (float)(m_total_length - m_input_read_pos) / (float)m_sample_rate;
	m_mutex_progress.unlock();
	return remaining;
}

void PCMPlayer::draw()
{
	int width, height;
	glfwGetFramebufferSize(m_window, &width, &height);
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);

	switch (m_mode)
	{
	case Mode::WaveForm:
	{
		int start_buf = (m_next_buf + 10) % 12;
		for (int i = 0; i < 2; i++)
		{
			int buf_i = (start_buf + i) % 12;
			for (int j = 0; j < AUDIO_BUF_LEN; j++)
			{
				int k = i * AUDIO_BUF_LEN + j;
				m_wavaform_cache[k] = ((float)m_ring_buf[buf_i].data[j * 2] + (float)m_ring_buf[buf_i].data[j * 2 + 1]) / 65536.0f;
			}
		}

		glColor3f(1.0f, 1.0f, 0.0f);
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < AUDIO_BUF_LEN * 2; i++)
		{
			float x = (float)i / (float)(AUDIO_BUF_LEN * 2 - 1)*2.0f - 1.0f;
			float y = m_wavaform_cache[i]*0.75f;
			glVertex2f(x, y);
		}		
		glEnd();
	}
	break;
	case Mode::Spectrum:
	{
		int start_buf = (m_next_buf + 8) % 12;
		for (int i = 0; i < 4; i++)
		{
			int buf_i = (start_buf + i) % 12;
			for (int j = 0; j < AUDIO_BUF_LEN; j++)
			{
				int k = i * AUDIO_BUF_LEN + j;
				m_spectrum_cache[k] = ((float)m_ring_buf[buf_i].data[j * 2] + (float)m_ring_buf[buf_i].data[j * 2 + 1]) / 65536.0f;
			}
		}

		for (unsigned i = 0; i < 2048; i++)
		{
			float x = (float)((int)i - 1024) / 1024.0f;
			float win = 0.5f*(cosf(x*(float)PI) + 1.0f);
			m_fftData[i].Re = win * m_spectrum_cache[i];
			m_fftData[i].Im = 0.0f;
		}

		fft(m_fftData, 11);

		for (unsigned i = 0; i < 100; i++)
		{
			float fstart = powf(2.0f, (float)i*0.1f);
			float fstop = powf(2.0f, (float)(i + 1)*0.1f);

			unsigned ustart = (unsigned)ceilf(fstart);
			unsigned ustop = (unsigned)ceilf(fstop);

			float ave = 0.0f;
			if (ustart == ustop)
				ave = DCAbs(&m_fftData[ustart]);
			else
			{
				for (unsigned j = ustart; j < ustop; j++)
				{
					ave += DCAbs(&m_fftData[j]);
				}
				ave /= (float)(ustop - ustart);
			}

			m_barv[i] = logf(ave*10.0f) / 10.0f;

		}

		glBegin(GL_QUADS);

		for (unsigned i = 0; i < 100.0f; i++)
		{
			float center = ((float)i + 0.5f) / 100.0f;
			float halfWidth = 0.4f / 100.0f;
			float left = center - halfWidth;
			float right = center + halfWidth;

			left = left * 2.0f - 1.0f;
			right = right * 2.0f - 1.0f;

			float bottom = -1.0f;

			float v = m_barv[i];
			if (v > 1.0f) v = 1.0f;
			float top = v * 2.0f - 1.0f;

			glColor3f(0.5f, 1.0f, 0.0f);
			glVertex2f(right, bottom);
			glVertex2f(left, bottom);

			glColor3f(0.5f + 0.5f*v, 1.0f - v, 0.0f);
			glVertex2f(left, top);
			glVertex2f(right, top);
		}
		glEnd();

	}
	break;
	}

}

void PCMPlayer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	PCMPlayer* self = (PCMPlayer*)glfwGetWindowUserPointer(window);
	if (key == GLFW_KEY_W && action == GLFW_PRESS)
	{
		self->m_mode = Mode::WaveForm;
	}

	if (key == GLFW_KEY_S && action == GLFW_PRESS)
	{
		self->m_mode = Mode::Spectrum;
	}

}

void PCMPlayer::main_loop()
{
	if (m_window == nullptr) return;
	while (glfwWindowShouldClose(m_window) == 0)
	{
		glfwMakeContextCurrent(m_window);
		draw();
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
}
