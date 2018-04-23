def getCV(CVLyric):
	vowels= ["a","e","i","o","u","v"]
	min_i=len(CVLyric)
	for c in vowels:
		i=CVLyric.find(c)
		if i>-1 and i<min_i:
			min_i=i

	consonant= CVLyric[0:min_i]
	vowel=CVLyric[min_i:len(CVLyric)]

	if CVLyric=="zhi" or CVLyric=="chi" or CVLyric=="shi" or CVLyric=="ri":
		vowel="h-i"
	if CVLyric=="zi" or CVLyric=="ci" or CVLyric=="si":
		vowel="-i"
	if CVLyric=="ju" or CVLyric=="qu" or CVLyric=="xu" or CVLyric=="yu":
		vowel="v"
	if CVLyric=="ye":
		vowel="eh"
	if CVLyric=="lv":
		CVLyric="lyu"
	if CVLyric=="nv":
		CVLyric="nyu"

	if vowel=="ia":
		vowel="ya"
	if vowel=="iao":
		vowel="yao"
	if vowel=="ian":
		vowel="yan"
	if vowel=="iang":
		vowel="yang"
	if vowel=="ie":
		vowel="eh"
	if vowel=="in":
		vowel="en"
	if vowel=="ing":
		vowel="eng"
	if vowel=="ong":
		vowel="weng"
	if vowel=="iong":
		vowel="weng"
	if vowel=="iu":
		vowel="you"
	if vowel=="ua":
		vowel="wa"
	if vowel=="uai":
		vowel="wai"
	if vowel=="uan":
		vowel="wan"
	if vowel=="uai":
		vowel="wai"
	if vowel=="ui":
		vowel="wei"
	if vowel=="uang":
		vowel="wang"
	if vowel=="un":
		vowel="wen"
	if vowel=="uo":
		vowel="wo"
	if (vowel=="ue"):
		vowel="ueh"

	if vowel=="i":
		vowel="y"
	if vowel=="u":
		vowel="w"
	if vowel=="v":
		vowel="yu"

	if consonant=="":
		if vowel=="a" or vowel=="ai" or vowel=="an":
			consonant="a"
		if vowel=="er" or vowel=="ao" or vowel=="ang":
			consonant="ah"
		if vowel=="en" or vowel=="eng":
			consonant="en"
		if vowel=="u":
			consonant="w"
		if vowel=="y":
			consonant="y"
		if vowel=='yu':
			consonant="yu"

	return (consonant,vowel, CVLyric)

# v1
def XiaYYConverter(LyricForEachSyllable):	
	CV = [getCV(lyric) for lyric in  LyricForEachSyllable]
	ret=[]
	for i in range(len(LyricForEachSyllable)):
		lyric=CV[i][2]
		if i==0:
			lyric='- '+lyric
		elif CV[i][0]=='':
			lyric=CV[i-1][1]+" "+lyric
		else:
			lyric+='*'

		if i<len(LyricForEachSyllable)-1 and CV[i+1][0]!='':
			if (CV[i][1]==CV[i+1][0]):
				ret+=[(lyric,1.0,True)]
			else:
				ret+=[(lyric,0.75,True, CV[i][1]+" "+CV[i+1][0], 0.25,False)]

		else:
			if CV[i][1]=='ai' or CV[i][1]=='ei' or CV[i][1]=='wai' or CV[i][1]=='wei':
				ret+=[(lyric,0.875, True, CV[i][1]+" y", 0.125, False)]
			elif CV[i][1]=='ou' or CV[i][1]=='you':
				ret+=[(lyric,0.875, True, CV[i][1]+" w", 0.125, False)]
			else:
				ret+=[(lyric,1.0,True)]
	return ret

