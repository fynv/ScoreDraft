def getVowel(CVLyric):
	vowels= ["a","e","i","o","u","v"]
	min_i=len(CVLyric)
	for c in vowels:
		i=CVLyric.find(c)
		if i>-1 and i<min_i:
			min_i=i
	vowel=CVLyric[min_i:len(CVLyric)]

	if CVLyric=="zhi" or CVLyric=="chi" or CVLyric=="shi" or CVLyric=="ri":
		vowel="ir"
	if CVLyric=="zi" or CVLyric=="ci" or CVLyric=="si":
		vowel="iz"
	if CVLyric=="ju" or CVLyric=="qu" or CVLyric=="xu" or CVLyric=="yu":
		vowel="v"
	if CVLyric=="ye":
		vowel="ie"

	if vowel=="ia":
		vowel="a"
	if vowel=="iao":
		vowel="ao"
	if vowel=="ian":
		vowel="an"
	if vowel=="iang":
		vowel="ang"
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
	return vowel

def TsuroVCVConverter(LyricForEachSyllable):
	vowels= [getVowel(lyric) for lyric in  LyricForEachSyllable]
	ret=[]
	for i in range(len(LyricForEachSyllable)):
		v='-'
		if i>0:
			v= vowels[i-1]
		ret+=[(v+' '+LyricForEachSyllable[i], 1.0, True)]
	return ret


