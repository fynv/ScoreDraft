def pinyinCVParser(CVLyric):
	vowels= ["a","e","i","o","u","v"]
	min_i=len(CVLyric)
	for c in vowels:
		i=CVLyric.find(c)
		if i>-1 and i<min_i:
			min_i=i
	consonant= CVLyric[0:min_i]
	vowel=CVLyric[min_i:len(CVLyric)]
	if vowel=="i" and (consonant=="zh" or consonant=="ch" or consonant=="sh" or consonant=="r"):
		vowel="ir"
	if vowel=="i" and (consonant=="z" or consonant=="c" or consonant=="s"):
		vowel="iz"
	if vowel=="u" and (consonant=="j" or consonant=="q" or consonant=="x" or consonant=="y"):
		vowel="v"
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
	if CVLyric=="yi" or CVLyric=="wu" or CVLyric=="yu":
		consonant=""
	return (consonant,vowel,CVLyric,"")
