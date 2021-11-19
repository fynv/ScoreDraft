#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

typedef std::unordered_map<unsigned, unsigned char*> ColorMap;

struct Pos2D
{
	float x;
	float y;
};

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
class AudioBufferQueue;

struct PaStreamCallbackTimeInfo;

struct GLFWwindow;

class Meteor;
struct VisSinging;
class MeteorPlayer
{
public:
	MeteorPlayer(Meteor* meteor, TrackBuffer* trackbuffer);
	~MeteorPlayer();
	
	void main_loop();	

private:
	static thread_local int s_instance_count;
	void *m_stream;
	static int stream_callback(const void *inputBuffer, void *outputBuffer,
		unsigned long framesPerBuffer,
		const PaStreamCallbackTimeInfo* timeInfo,
		unsigned long statusFlags,
		void *userData);

	double m_sample_rate = 44100.0;
	Meteor* m_meteor = nullptr;
	TrackBuffer* m_trackbuffer = nullptr;
	std::unique_ptr<AudioBufferQueue> m_audio_queue;
	size_t m_input_read_pos = 0;
	size_t m_audio_read_pos = 0;
	
	size_t m_playback_pos = 0;
	size_t m_playback_pos_next = 0;
	double m_time_playback_pos = 0.0;	
	std::mutex m_mutex_sync;
	void _set_sync_point(size_t playback_pos, double time_playback_pos);
	void _get_sync_point(size_t& playback_pos, double& time_playback_pos);

	bool m_demuxing = false;
	static void thread_demux(MeteorPlayer* self);
	std::unique_ptr<std::thread> m_thread_demux;

	// UI
	GLFWwindow* m_window = nullptr;
	void draw();

	struct TexText
	{
		int width;
		int height;
		unsigned texId;
	};

	std::unordered_map<std::string, TexText> m_texs;
	void _buildTexs();
	void _draw_tex(const TexText& tex, float x, float y);

	void _buildColorMap();
	void _draw_key(float left, float right, float bottom, float top, float lineWidth, bool black = false);
	void _draw_flash(float centerx, float centery, float radius, unsigned char color[3], float alpha);
	static unsigned char s_ColorBank[15][3];

	ColorMap m_InstColorMap;
	ColorMap m_PercColorMap;
	std::vector<Pos2D> m_beats_centers;
	ColorMap m_SingerColorMap;

	// UI settings
	float m_whiteKeyWidth = 18.0f;
	float m_blackKeyWidth = 14.0f;
	float m_whiteKeyHeight = 80.0f;
	float m_blackKeyHeight = 50.0f;
	float m_cornerSize = 3.0f;
	float m_whiteKeyPressedDelta = 3.0f;
	float m_blackKeyPressedDelta = 2.0f;
	float m_pressedLineWidth = 3.0f;

	float m_showTime = 1.0f;
	float m_meteorHalfWidth = 5.0f;

	float m_percussion_flash_size_factor = 0.15f;
	float m_percussion_flash_limit = 0.3f;

	float m_singing_half_width = 8.0f;
};