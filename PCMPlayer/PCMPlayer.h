#pragma once

#include <memory>
#include <mutex>
#include "fft.h"

namespace std
{
	class thread;
}

#define AUDIO_BUF_LEN 512
struct AudioBuffer
{
	short data[AUDIO_BUF_LEN * 2];
};

class TrackBuffer;
class TrackBufferQueue;
class AudioBufferQueue;

struct PaStreamCallbackTimeInfo;

struct GLFWwindow;


class PCMPlayer
{
public:
	PCMPlayer(double sample_rate = 44100.0, bool ui = false);
	~PCMPlayer();

	void PlayTrack(TrackBuffer &track);
	float GetRemainingTime();

	// UI
	void main_loop();

private:
	static thread_local int s_instance_count;	
	void *m_stream;
	static int stream_callback(const void *inputBuffer, void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		unsigned long statusFlags,
		void *userData);

	double m_sample_rate;
	std::unique_ptr<TrackBufferQueue> m_input_queue;
	std::unique_ptr<AudioBufferQueue> m_audio_queue;
	size_t m_input_read_pos = 0;
	size_t m_audio_read_pos = 0;
	size_t m_total_length = 0;

	std::mutex m_mutex_progress;

	bool m_demuxing = false;
	static void thread_demux(PCMPlayer* self);
	std::unique_ptr<std::thread> m_thread_demux;

	// UI
	GLFWwindow* m_window = nullptr;
	void draw();
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	enum class Mode
	{
		Spectrum,
		WaveForm
	};

	Mode m_mode = Mode::WaveForm;

	int m_next_buf = 0;
	AudioBuffer m_ring_buf[12];
	float m_wavaform_cache[AUDIO_BUF_LEN * 2];
	float m_spectrum_cache[AUDIO_BUF_LEN * 4];
	DComp m_fftData[2048];
	float m_barv[100];

};