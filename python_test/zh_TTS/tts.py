import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
from ScoreDraftRapChinese import *

# https://github.com/mozillazg/python-pinyin
# pip install pypinyin
from pypinyin import pinyin, Style

import re
import sys

singer = ScoreDraft.GePing_UTAU()
'''
import CVVCChineseConverter
singer = ScoreDraft.YuMo_UTAU()
ScoreDraft.UtauDraftSetLyricConverter(singer, CVVCChineseConverter.CVVCChineseConverter)
'''

tempo = 100
ref_freq=264.0

player=ScoreDraft.QPlayTrackBuffer

def TTS(sentence):
	pinyins = pinyin(sentence,style=Style.TONE3)
	splits=[]	
	for word in pinyins:
		splitted=re.findall(r'[0-9]+|[a-z]+|[\S&^0-9&^a-z]+', word[0])
		if len(splitted)<2:
			if sentence.find(splitted[0])<0:
				splitted+=['5']
		elif not splitted[1].isdigit():
			splitted[:]=splitted[0:0]
		splits+=[splitted]

	# apply rules:
	for i in reversed(range(len(splits)-1)):
		cur_word = splits[i]
		next_word = splits[i+1]
		if len(cur_word)>1 and len(next_word)>1:
			if int(cur_word[1])==3 and int(next_word[1])==3:
				cur_word[1]=2

	rap_seq=[]
	sub_seq=()
	for splitted in splits:
		if len(splitted)<2:
			if len(sub_seq)>0:
				rap_seq+=[sub_seq, BL(24)]
				sub_seq=()
		else:
			sub_seq+=CRap(splitted[0], int(splitted[1]), 24)

	if len(sub_seq)>0:
		rap_seq+=[sub_seq, BL(24)]
		sub_seq=()

	if len(rap_seq)>0:
		buf=ScoreDraft.TrackBuffer(1)
		singer.sing(buf, rap_seq, tempo, ref_freq)
		return buf

	return None

def TTS_play(sentence):
	buf=TTS(sentence)
	if buf != None:
		player(buf)
	
import os
if __name__ == '__main__':
	if (len(sys.argv)>1):
		with open(sys.argv[1],'r') as f:
			while True:
				line = f.readline()
				if not line:
					break
				TTS_play(line.strip('\n'))
		exit(0)
	print("输入点什么来朗读吧：")
	if os.name == 'nt':
		player=ScoreDraft.PlayTrackBuffer
	while True:
		line=sys.stdin.readline().strip('\n')
		TTS_play(line)
