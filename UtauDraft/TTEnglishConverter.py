import os
import pickle
ScoreDraftPath= os.path.dirname(__file__)

lyricSet=set()
lyricPrefixSet=set()
vowelSet={'eI','aI','aU','OI','oU','3','i','I','U','u','E','{','A','V','O','@'}
vowelPrefixSet=set()

atomicSet={'tS','dZ','dr','tr'}

def BuildLyricSet():
	with open(ScoreDraftPath+'/UTAUVoice/TetoEng/Teto/oto.ini', 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			p1 = line.find('=')
			if p1==-1:
				continue
			fn=line[0:p1-4]
			p2 = line.find(',',p1)
			if p2==-1:
				continue
			lyric=line[p1+1:p2]
			if lyric=='':
				lyric=fn
			lyricSet.add(lyric)
	with open(ScoreDraftPath+'/TTLyricSet.data','wb') as f:
		pickle.dump(lyricSet,f)

def LoadLyricSet():
	with open(ScoreDraftPath+'/TTLyricSet.data','rb') as f:
		global lyricSet
		lyricSet=pickle.load(f)

#BuildLyricSet()
LoadLyricSet()

for lyric in lyricSet:
	for i in range(len(lyric)):
		if lyric[i]==' ':
			continue
		lyricPrefixSet.add(lyric[0:i+1])

for vowel in vowelSet:
	for i in range(len(vowel)):
		vowelPrefixSet.add(vowel[0:i+1])

def TTEnglishConverter(inList):
	inList_a=[]
	for inLyric in inList:
		lyric_a=[]
		i=0
		while i<len(inLyric):
			atom=''
			while i<len(inLyric) and (atom=='' or (atom+inLyric[i] in atomicSet)):
				atom=atom+inLyric[i]
				i+=1
			lyric_a+=[atom]
		inList_a+=[lyric_a]

	inList_a[len(inList_a)-1]+=['-']

	vowelMap=[]

	for inLyric in inList_a:
		start=-1
		end=-1
		for i in range(len(inLyric)):
			if start==-1:
				if inLyric[i] in vowelPrefixSet:
					start=i
			if start!=-1 and (''.join(inLyric[start:i+1]) in vowelPrefixSet):
				end=i+1
		vowelMap+=[(start,end)]

	cur=[0,0]
	outList=[]

	prefix='-'
	iIn=0

	while cur[0]<len(inList_a) and cur[1]<len(inList_a[cur[0]]):	
		# pass 1 
		while len(prefix)>0:
			if prefix=='-' or cur[1]>0 or vowelMap[cur[0]][0]==0:
				test_seg = prefix + inList_a[cur[0]][cur[1]]
				if test_seg in lyricPrefixSet:
					break

			test_seg = prefix +' '+inList_a[cur[0]][cur[1]]
			if test_seg in lyricPrefixSet:
				break

			prefix=prefix[1:len(prefix)]

		#pass2
		nextStart=cur[:]
		seg=''
		isVowel=False

		while True:
			seg=''
			lastSeg=prefix[:]
			cur2=cur[:] 

			isVowel=False

			while True:
				spaceMust=False
				newChar=''
				if not (cur2[0]<len(inList_a) and cur2[1]<len(inList_a[cur2[0]])):
					newChar='-'
				else:
					newChar= inList_a[cur2[0]][cur2[1]]
					if lastSeg!='' and lastSeg!='-' and cur2[1]==0 and vowelMap[cur2[0]][0]>0:
						spaceMust=True

				if lastSeg=='' and cur[0]<len(inList_a) and cur[1]>=vowelMap[cur[0]][0] and cur[1]<vowelMap[cur[0]][1]:
					lastSeg='-'

				test_seg=lastSeg+newChar
				if spaceMust or not (test_seg in lyricPrefixSet):
					test_seg=lastSeg+' '+newChar
					if not (test_seg in lyricPrefixSet):
						break

				lastSeg=test_seg

				if test_seg in lyricSet:
					cur=cur2[:]
					seg=test_seg
					if cur[0]<len(inList_a):
						if cur[1]>=vowelMap[cur[0]][0] and cur[1]<vowelMap[cur[0]][1]:
							isVowel=True
							iIn=cur[0]

				if not (cur2[0]<len(inList_a) and cur2[1]<len(inList_a[cur2[0]])):
					break

				cur2[1]+=1
				if cur2[1]>=len(inList_a[cur2[0]]):
					cur2[0]+=1
					cur2[1]=0	
					if seg!='':
						break			


			if len(seg)>0 or len(prefix)==0:
				break
			
			prefix=prefix[1:len(prefix)]	

		if len(seg)>0:
			outList+=[(seg, iIn, isVowel)]

		if not (cur[0]<len(inList_a) and cur[1]<len(inList_a[cur[0]])):
			break

		cur[1]+=1
		if cur[1]>=len(inList_a[cur[0]]):
			cur[0]+=1
			cur[1]=0

		pos=nextStart[:]
		prefix=''
		while pos[0]<cur[0] or (pos[0]==cur[0] and pos[1]<cur[1]):
			prefix+=inList_a[pos[0]][pos[1]]
			pos[1]+=1
			if pos[1]>=len(inList_a[pos[0]]):
				pos[0]+=1
				pos[1]=0

	ret=[]
	syllable=()
	iSyllable=0

	for i in range(len(outList)):
		outItem=outList[i]
		if outItem[1]!=iSyllable:
			ret+=[syllable]
			syllable=()
			iSyllable=outItem[1]
		weight=0.1
		if outItem[2]:
			weight=0.4
		syllable+=(outItem[0], weight, outItem[2])

	ret+=[syllable]

	#print(ret)

	return ret
