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
		vowel="ir"
	if CVLyric=="zi" or CVLyric=="ci" or CVLyric=="si":
		vowel="i0"
	if CVLyric=="ju" or CVLyric=="qu" or CVLyric=="xu" or CVLyric=="yu":
		vowel="v"
	if CVLyric=="ye":
		vowel="e0"

	if vowel=="ia":
		vowel="a"
	if vowel=="iao":
		vowel="ao"
	if vowel=="ian":
		vowel="an"
	if vowel=="iang":
		vowel="ang"
	if vowel=="ie":
		vowel="e0"
	if vowel=="iong":
		vowel="ong"
	if vowel=="iu":
		vowel="ou"
	if vowel=="ua":
		vowel="a"
	if vowel=="uai":
		vowel="ai"
	if vowel=="uan":
		vowel="an"
	if vowel=="uai":
		vowel="ai"
	if vowel=="ui":
		vowel="ei"
	if vowel=="uang":
		vowel="ang"
	if vowel=="un":
		vowel="en"
	if vowel=="uo":
		vowel="o"
	if (vowel=="ue"):
		vowel="e0"

	if consonant=="j" or consonant=="q" or consonant=="x":
		if vowel[0]=="u":
			consonant+="w"
		else:
			consonant+="y"

	if consonant=="y":
		if vowel[0]=="u":
			consonant="v"
		else:
			consonant="y"
	return (consonant,vowel)

# v1
def CVVCChineseConverter(LyricForEachSyllable):	
	CV = [getCV(lyric) for lyric in  LyricForEachSyllable]
	ret=[]
	for i in range(len(LyricForEachSyllable)):
		lyric=LyricForEachSyllable[i]
		if i==0:
			lyric='- '+lyric
		elif CV[i][0]=='':
			lyric=CV[i-1][1]+" "+lyric
		if i<len(LyricForEachSyllable)-1:
			if CV[i+1][0]!='':
				ret+=[(lyric,0.875, True, CV[i][1]+" "+CV[i+1][0], 0.125, False)]
			else:
				ret+=[(lyric,1.0, True)]
		else:
			ret+=[(lyric,0.875, True, CV[i][1]+" R", 0.125, False)]
	return ret
