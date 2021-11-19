#include <thread>
#include <mutex>
#include <vector>
#include <list>
#include <queue>
#include <unordered_set>
#include <memory.h>
#include <cmath>
#include <portaudio.h>
#include <GLFW/glfw3.h>

#include <TrackBuffer.h>
#include <utils.h>
#include "MeteorPlayer.h"
#include "Meteor.h"
#include "DrawText.h"

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


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


void MeteorPlayer::_set_sync_point(size_t playback_pos, double time_playback_pos)
{
	m_mutex_sync.lock();
	m_playback_pos = playback_pos;
	m_time_playback_pos = time_playback_pos;
	m_mutex_sync.unlock();
}

void MeteorPlayer::_get_sync_point(size_t& playback_pos, double& time_playback_pos)
{
	m_mutex_sync.lock();
	playback_pos = m_playback_pos;
	time_playback_pos = m_time_playback_pos;
	m_mutex_sync.unlock();
}


int MeteorPlayer::stream_callback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	unsigned long statusFlags,
	void *userData)
{
	MeteorPlayer* self = (MeteorPlayer*)userData;
	AudioBufferQueue* queue = self->m_audio_queue.get();

	self->_set_sync_point(self->m_playback_pos_next, time_sec());

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

	self->m_playback_pos_next = self->m_playback_pos + framesPerBuffer;

	return paContinue;
}

void MeteorPlayer::thread_demux(MeteorPlayer* self)
{
	TrackBuffer* buf_in = self->m_trackbuffer;
	unsigned size = buf_in->NumberOfSamples();
	unsigned chn = buf_in->NumberOfChannels();
	float volume = buf_in->AbsoluteVolume();
	float pan = buf_in->Pan();

	AudioBufferQueue* queue_out = self->m_audio_queue.get();
	while (self->m_demuxing)
	{
		AudioBuffer* buf_out = new AudioBuffer;
		size_t pos_out = 0;
		while (pos_out < AUDIO_BUF_LEN)
		{
			size_t copy_size = min(size - self->m_input_read_pos, AUDIO_BUF_LEN - pos_out);
			for (size_t i = 0; i < copy_size; i++)
			{
				size_t read_pos = self->m_input_read_pos + i;
				float sample_in[2];
				buf_in->Sample(read_pos, sample_in);
				short sample_out[2];
				if (chn == 1)
				{
					sample_out[0] = sample_out[1] = (short)(max(min(sample_in[0] * volume, 1.0f), -1.0f)*32767.0f);
				}
				else
				{
					CalcPan(pan, sample_in[0], sample_in[1]);
					sample_out[0] = (short)(max(min(sample_in[0] * volume, 1.0f), -1.0f)*32767.0f);
					sample_out[1] = (short)(max(min(sample_in[1] * volume, 1.0f), -1.0f)*32767.0f);
				}

				size_t write_pos = pos_out + i;
				buf_out->data[write_pos * 2] = sample_out[0];
				buf_out->data[write_pos * 2 + 1] = sample_out[1];

			}			
			self->m_input_read_pos += copy_size;			
			pos_out += copy_size;
			if (self->m_input_read_pos == size)
			{
				self->m_demuxing = false;
				break;
			}
		}
		if (pos_out < AUDIO_BUF_LEN)
			memset(buf_out->data + pos_out * 2, 0, sizeof(short) * (AUDIO_BUF_LEN - pos_out) * 2);

		queue_out->Push(buf_out);
	}	
}


#define PI 3.1415926535897932384626433832795

inline float rand01()
{
	float f = (float)rand() / (float)RAND_MAX;
	if (f < 0.0000001f) f = 0.0000001f;
	if (f > 0.9999999f) f = 0.9999999f;
	return f;
}


unsigned char MeteorPlayer::s_ColorBank[15][3] =
{
	{ 0x41, 0x8C, 0xF0 },
	{ 0xFC, 0xB4, 0x41 },
	{ 0xDF, 0x3A, 0x02 },
	{ 0x05, 0x64, 0x92 },
	{ 0xBF, 0xBF, 0xBF },
	{ 0x1A, 0x3B, 0x69 },
	{ 0xFF, 0xE3, 0x82 },
	{ 0x12, 0x9C, 0xDD },
	{ 0xCA, 0x6B, 0x4B },
	{ 0x00, 0x5C, 0xDB },
	{ 0xF3, 0xD2, 0x88 },
	{ 0x50, 0x63, 0x81 },
	{ 0xF1, 0xB9, 0xA8 },
	{ 0xE0, 0x83, 0x0A },
	{ 0x78, 0x93, 0xBE }
};



thread_local int MeteorPlayer::s_instance_count = 0;
MeteorPlayer::MeteorPlayer(Meteor* meteor, TrackBuffer* trackbuffer)
	: m_meteor(meteor), m_trackbuffer(trackbuffer)
{
	m_time_playback_pos = time_sec();

	if (s_instance_count == 0) Pa_Initialize();
	s_instance_count++;

	m_audio_queue = std::unique_ptr<AudioBufferQueue>(new AudioBufferQueue(8));

	m_demuxing = true;
	m_thread_demux = (std::unique_ptr<std::thread>)(new std::thread(thread_demux, this));

	PaStreamParameters outputParameters = {};
	outputParameters.device = Pa_GetDefaultOutputDevice();
	outputParameters.channelCount = 2;
	outputParameters.sampleFormat = paInt16;
	outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;

	double sample_rate = (double)trackbuffer->Rate();
	Pa_OpenStream(&m_stream, nullptr, &outputParameters, sample_rate, (unsigned long)(sample_rate*0.01), paClipOff, stream_callback, this);
	Pa_StartStream(m_stream);

	glfwInit();
	glfwWindowHint(GLFW_SAMPLES, 4);
	m_window = glfwCreateWindow(640, 480, "ScoreDraft Meteor Player", NULL, NULL);
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1);

	_buildColorMap();
	_buildTexs();
}

MeteorPlayer::~MeteorPlayer()
{
	auto iter = m_texs.begin();
	while (iter != m_texs.end())
	{
		glDeleteTextures(1, &iter->second.texId);
		iter++;
	}

	glfwDestroyWindow(m_window);
	glfwTerminate();

	m_demuxing = false;

	Pa_StopStream(m_stream);
	Pa_CloseStream(m_stream);

	while (m_audio_queue->Size() > 0)
	{
		AudioBuffer* p = m_audio_queue->Pop();
		delete p;
	}

	m_thread_demux->join();
	m_thread_demux = nullptr;

	s_instance_count--;
	if (s_instance_count == 0) Pa_Terminate();
}


void MeteorPlayer::_buildColorMap()
{
	unsigned bankRef = 0;
	const std::vector<VisBeat>&  beats = m_meteor->GetBeats();
	m_beats_centers.clear();
	for (unsigned i = 0; i < (unsigned)beats.size(); i++)
	{
		unsigned perc = beats[i].percId;
		if (m_PercColorMap.find(perc) == m_PercColorMap.end())
		{
			m_PercColorMap[perc] = s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= 15) bankRef = 0;
		}

		float x = rand01();
		float y = rand01();

		m_beats_centers.push_back({ x, y });
	}

	const std::vector<VisSinging>&  singings = m_meteor->GetSingings();
	for (unsigned i = 0; i < (unsigned)singings.size(); i++)
	{
		unsigned singer = singings[i].singerId;
		if (m_SingerColorMap.find(singer) == m_SingerColorMap.end())
		{
			m_SingerColorMap[singer] = s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= 15) bankRef = 0;
		}
	}

	const std::vector<VisNote>&  notes = m_meteor->GetNotes();
	for (unsigned i = 0; i < (unsigned)notes.size(); i++)
	{
		unsigned inst = notes[i].instrumentId;
		if (m_InstColorMap.find(inst) == m_InstColorMap.end())
		{
			m_InstColorMap[inst] = s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= 15) bankRef = 0;
		}
	}
}

void MeteorPlayer::_buildTexs()
{
	const std::vector<VisSinging>& lst_singings = m_meteor->GetSingings();
	for (size_t i = 0; i < lst_singings.size(); i++)
	{
		const VisSinging* pnote = &lst_singings[i];
		std::string lyric = pnote->lyric;
		Text text(lyric.c_str());
		TexText tex;
		tex.width = text.m_width;
		tex.height = text.m_height;
		glGenTextures(1, &tex.texId);
		glBindTexture(GL_TEXTURE_2D, tex.texId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.width, tex.height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, text.m_data.data());
		glBindTexture(GL_TEXTURE_2D, 0);
		m_texs[lyric] = tex;
	}

}

void MeteorPlayer::_draw_tex(const TexText& tex, float x, float y)
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex.texId);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(x, y);

	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(x + tex.width, y);

	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(x + tex.width, y + tex.height);

	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(x, y + tex.height);
	glEnd();

	glDisable(GL_TEXTURE_2D);
}


void MeteorPlayer::_draw_key(float left, float right, float bottom, float top, float lineWidth, bool black)
{
	if (black)
		glColor3f(0.0f, 0.0f, 0.0f);
	else
		glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_QUADS);
	// top
	glVertex2f(left + m_cornerSize, top);
	glVertex2f(right - m_cornerSize, top);
	glVertex2f(right, top - m_cornerSize);
	glVertex2f(left, top - m_cornerSize);
	// mid
	glVertex2f(left, top - m_cornerSize);
	glVertex2f(right, top - m_cornerSize);
	glVertex2f(right, bottom + m_cornerSize);
	glVertex2f(left, bottom + m_cornerSize);
	// bottom
	glVertex2f(left, bottom + m_cornerSize);
	glVertex2f(right, bottom + m_cornerSize);
	glVertex2f(right - m_cornerSize, bottom);
	glVertex2f(left + m_cornerSize, bottom);
	glEnd();

	// outline
	if (black)
		glColor3f(1.0f, 1.0f, 1.0f);
	else
		glColor3f(0.0f, 0.0f, 0.0f);
	glLineWidth(lineWidth);
	glBegin(GL_LINE_STRIP);
	glVertex2f(left + m_cornerSize, top);
	glVertex2f(right - m_cornerSize, top);
	glVertex2f(right, top - m_cornerSize);
	glVertex2f(right, bottom + m_cornerSize);
	glVertex2f(right - m_cornerSize, bottom);
	glVertex2f(left + m_cornerSize, bottom);
	glVertex2f(left, bottom + m_cornerSize);
	glVertex2f(left, top - m_cornerSize);
	glVertex2f(left + m_cornerSize, top);
	glEnd();

}


void MeteorPlayer::_draw_flash(float centerx, float centery, float radius, unsigned char color[3], float alpha)
{
	unsigned div = 36;
	unsigned char uAlpha = (unsigned char)(alpha*255.0f);

	glBegin(GL_TRIANGLES);
	for (unsigned i = 0; i < div; i++)
	{
		float theta1 = (float)i / (float)div * 2.0f*(float)PI;
		float theta2 = (float)(i + 1) / (float)div * 2.0f*(float)PI;

		float x1 = centerx + cosf(theta1)*radius;
		float y1 = centery + sinf(theta1)*radius;

		float x2 = centerx + cosf(theta2)*radius;
		float y2 = centery + sinf(theta2)*radius;

		glColor4ub(color[0], color[1], color[2], uAlpha);
		glVertex2f(centerx, centery);
		glColor4ub(color[0], color[1], color[2], 0);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
	}

	glEnd();
}


void MeteorPlayer::draw()
{
	int width, height;
	glfwGetFramebufferSize(m_window, &width, &height);
	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, (double)width, 0.0, (double)height, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);

	const SubLists& notes_sublists = m_meteor->GetNotesSublists();
	const SubLists& beats_sublists = m_meteor->GetBeatsSublists();
	const SubLists& singing_sublists = m_meteor->GetSingingSublists();

	size_t playback_pos;
	double time_playback_pos;
	_get_sync_point(playback_pos, time_playback_pos);
	double ref_time = (double)playback_pos / (double)m_trackbuffer->Rate();
	double time_now = time_sec();

	float note_inTime = (float)(ref_time + (time_now - time_playback_pos));
	unsigned note_intervalId = notes_sublists.GetIntervalId(note_inTime);
	float note_outTime = note_inTime - m_showTime;
	unsigned note_intervalId_min = notes_sublists.GetIntervalId(note_outTime);		
	
	/// draw meteors
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	//notes
	if (notes_sublists.m_subLists.size() > 0)
	{
		std::unordered_set<const VisNote*> visiableNotes;

		for (unsigned i = note_intervalId_min; i <= note_intervalId; i++)
		{
			const SubList& subList = notes_sublists.m_subLists[i];
			for (unsigned j = 0; j < (unsigned)subList.size(); j++)
			{
				const VisNote& note = m_meteor->GetNotes()[subList[j]];
				if (note.start<note_inTime && note.end> note_outTime)
					visiableNotes.insert(&note);
			}
		}

		glBegin(GL_QUADS);

		float keyPos[12] = { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.5f, 4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f };

		for (auto iter = visiableNotes.begin(); iter != visiableNotes.end(); iter++)
		{
			float startY = ((*iter)->start - note_inTime) / -m_showTime * ((float)height - m_whiteKeyHeight) + m_whiteKeyHeight;
			float endY = ((*iter)->end - note_inTime) / -m_showTime * ((float)height - m_whiteKeyHeight) + m_whiteKeyHeight;

			unsigned instId = (*iter)->instrumentId;
			unsigned char* color = m_InstColorMap[instId];

			int pitch = (*iter)->pitch;
			int octave = 0;
			while (pitch < 0)
			{
				pitch += 12;
				octave--;
			}
			while (pitch >= 12)
			{
				pitch -= 12;
				octave++;
			}

			float x = (float)width*0.5f + ((float)octave*7.0f + keyPos[pitch])*m_whiteKeyWidth;

			glColor4ub(color[0], color[1], color[2], 255);
			glVertex2f(x, startY);
			glVertex2f(x + m_meteorHalfWidth, startY - m_meteorHalfWidth);
			glColor4ub(color[0], color[1], color[2], 0);
			glVertex2f(x, endY);
			glColor4ub(color[0], color[1], color[2], 255);
			glVertex2f(x - m_meteorHalfWidth, startY - m_meteorHalfWidth);

		}


		glEnd();
	}

	// beats
	if (beats_sublists.m_subLists.size() > 0)
	{
		unsigned beat_intervalId = beats_sublists.GetIntervalId(note_inTime);

		const SubList& subList = beats_sublists.m_subLists[beat_intervalId];
		for (unsigned i = 0; i < (unsigned)subList.size(); i++)
		{
			unsigned beatIndex = subList[i];
			const VisBeat& beat = m_meteor->GetBeats()[beatIndex];

			float start = beat.start;
			float end = beat.end;

			// limting percussion flash time
			if (end - start > m_percussion_flash_limit)
				end = start + m_percussion_flash_limit;

			if (note_inTime >= start && note_inTime <= end)
			{
				float centerx = m_beats_centers[beatIndex].x*width;
				float centery = m_beats_centers[beatIndex].y*(height - m_whiteKeyHeight) + m_whiteKeyHeight;
				float radius = width * m_percussion_flash_size_factor;

				unsigned char* color = m_PercColorMap[beat.percId];
				float alpha = (end - note_inTime) / (end - start);
				_draw_flash(centerx, centery, radius, color, alpha);
			}
		}
	}


	// singing
	if (singing_sublists.m_subLists.size() > 0)
	{
		unsigned singing_intervalId = singing_sublists.GetIntervalId(note_inTime);
		unsigned singing_intervalId_min = singing_sublists.GetIntervalId(note_outTime);

		std::unordered_set<const VisSinging*> visiableNotes;

		for (unsigned i = singing_intervalId_min; i <= singing_intervalId; i++)
		{
			const SubList& subList = singing_sublists.m_subLists[i];
			for (unsigned j = 0; j < (unsigned)subList.size(); j++)
			{
				const VisSinging& note = m_meteor->GetSingings()[subList[j]];
				if (note.start<note_inTime && note.end> note_outTime)
					visiableNotes.insert(&note);
			}
		}

		float pixelPerPitch = m_whiteKeyWidth * 7.0f / 12.0f;

		glBegin(GL_QUADS);
		
		for (auto iter = visiableNotes.begin(); iter != visiableNotes.end(); iter++)
		{
			float startY = ((*iter)->start - note_inTime) / -m_showTime * ((float)height - m_whiteKeyHeight) + m_whiteKeyHeight;
			float endY = ((*iter)->end - note_inTime) / -m_showTime * ((float)height - m_whiteKeyHeight) + m_whiteKeyHeight;

			unsigned singerId = (*iter)->singerId;
			unsigned char* color = m_SingerColorMap[singerId];

			const float* pitches = &(*iter)->pitch[0];
			unsigned num_pitches = (unsigned)(*iter)->pitch.size();

			for (unsigned i = 0; i < num_pitches - 1; i++)
			{
				float x1 = pitches[i] * pixelPerPitch + (float)width*0.5f + m_whiteKeyWidth * 0.5f;
				float x2 = pitches[i + 1] * pixelPerPitch + (float)width*0.5f + m_whiteKeyWidth * 0.5f;

				float k1 = (float)i / (float)(num_pitches - 1);
				float y1 = startY * (1.0f - k1) + endY * k1;

				float k2 = (float)(i + 1) / (float)(num_pitches - 1);
				float y2 = startY * (1.0f - k2) + endY * k2;

				glColor4ub(color[0], color[1], color[2], (unsigned char)((1.0f - k1)*255.0f));
				glVertex2f(x1 - m_singing_half_width, y1);
				glVertex2f(x1 + m_singing_half_width, y1);
				glVertex2f(x2 + m_singing_half_width, y2);
				glVertex2f(x2 - m_singing_half_width, y2);

			}

		}

		glEnd();
	}

	// lyrics
	if (singing_sublists.m_subLists.size() > 0)
	{
		unsigned singing_intervalId = singing_sublists.GetIntervalId(note_inTime);
		unsigned singing_intervalId_min = singing_sublists.GetIntervalId(note_outTime);

		std::unordered_set<const VisSinging*> visiableNotes;

		for (unsigned i = singing_intervalId_min; i <= singing_intervalId; i++)
		{
			const SubList& subList = singing_sublists.m_subLists[i];
			for (unsigned j = 0; j < (unsigned)subList.size(); j++)
			{
				const VisSinging& note = m_meteor->GetSingings()[subList[j]];
				if (note.start<note_inTime && note.start> note_outTime)
					visiableNotes.insert(&note);
			}
		}

		float pixelPerPitch = m_whiteKeyWidth * 7.0f / 12.0f;	

		for (auto iter = visiableNotes.begin(); iter != visiableNotes.end(); iter++)
		{
			float startY = ((*iter)->start - note_inTime) / -m_showTime * ((float)height - m_whiteKeyHeight) + m_whiteKeyHeight;

			unsigned singerId = (*iter)->singerId;
			unsigned char* color = m_SingerColorMap[singerId];

			float x = (*iter)->pitch[0] * pixelPerPitch + (float)width*0.5f + m_whiteKeyWidth * 0.5f + m_singing_half_width;
			std::string lyric = (*iter)->lyric;

			TexText& tex = m_texs[lyric];
			glColor3ub(color[0], color[1], color[2]);
			_draw_tex(tex, x, startY);
		}
	}


	/// draw keyboard
	glDisable(GL_BLEND);

	static int whitePitchs[7] = { 0, 2, 4, 5, 7, 9, 11 };
	static int blackPitchs[5] = { 1, 3, 6, 8, 10 };
	static int blackPos[5] = { 1, 2, 4, 5, 6 };

	float center = (float)width *0.5f;
	float octaveWidth = m_whiteKeyWidth * 7.0f;

	int minOctave = -(int)ceilf(center / octaveWidth);
	int maxOctave = (int)floorf(center / octaveWidth);
	int numKeys = (maxOctave - minOctave + 1) * 12;
	int indexShift = -minOctave * 12;

	bool* pressed = new bool[numKeys];
	memset(pressed, 0, sizeof(bool)* numKeys);

	// notes
	if (notes_sublists.m_subLists.size() > 0)
	{
		const SubList& subList = notes_sublists.m_subLists[note_intervalId];
		for (unsigned i = 0; i < (unsigned)subList.size(); i++)
		{
			const VisNote& note = m_meteor->GetNotes()[subList[i]];
			float start = note.start;
			float end = note.end;

			// early key-up movement
			end -= (end - start)*0.1f;

			if (note_inTime >= start && note_inTime <= end)
			{
				int index = note.pitch + indexShift;
				if (index >= 0 && index < numKeys)
				{				
					pressed[index] = true;
				}
			}
		}
	}


	for (int i = minOctave; center + (float)i*octaveWidth < width; i++)
	{
		float octaveLeft = center + (float)i*octaveWidth;
		for (int j = 0; j < 7; j++)
		{
			int index = whitePitchs[j] + i * 12 + indexShift;
			bool keyPressed = pressed[index];

			float left = octaveLeft + (float)j*m_whiteKeyWidth;
			float right = left + m_whiteKeyWidth;
			float bottom = keyPressed ? m_whiteKeyPressedDelta : 0.0f;
			float top = m_whiteKeyHeight;
			_draw_key(left, right, bottom, top, keyPressed ? m_pressedLineWidth : 1.0f);
		}
		for (int j = 0; j < 5; j++)
		{
			int index = blackPitchs[j] + i * 12 + indexShift;
			bool keyPressed = pressed[index];

			float keyCenter = octaveLeft + (float)blackPos[j] * m_whiteKeyWidth;
			float left = keyCenter - m_blackKeyWidth / 2.0f;
			float right = keyCenter + m_blackKeyWidth / 2.0f;

			float bottom = keyPressed ? m_whiteKeyHeight - m_blackKeyHeight + m_blackKeyPressedDelta : m_whiteKeyHeight - m_blackKeyHeight;
			float top = m_whiteKeyHeight;
			_draw_key(left, right, bottom, top, keyPressed ? m_pressedLineWidth : 1.0f, true);
		}
	}

	delete[] pressed;
}

void MeteorPlayer::main_loop()
{	
	while (m_demuxing && glfwWindowShouldClose(m_window) == 0)
	{
		glfwMakeContextCurrent(m_window);
		draw();
		glfwSwapBuffers(m_window);
		glfwPollEvents();
	}
}
