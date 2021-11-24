#(ly:set-option 'midi-extension "mid")
\version "2.18.2"
\score 
{	  
	\relative c''
	{
	  <<
	  \new Staff 
	  {
	  	\time 2/4
	  	r8 g8 c d
	  	e e16 e e d c8 
	  	d4 r8 d16 e
	  	d c c c b8 c 
	  	a4 r8 c16 d
	  	e e e e d8 c 
	  	d16 d d d c8 b
	  	c2 r8 g c d
	  	e e16 e e d c8 
	  	d4 r8 d16 e
	  	d c c c b8 c 
	  	a4 r8 c16 d
	  	e e e e d8 c 
	  	d16 d d d c8 b
	  	c2 r4 e8 d
	  	c16 c c c c8 d 
	  	d4 e8 d
	  	c16 c c c e8 f 
	  	d2
	  	r4 e16 f8.
	  	g16 g g g g f e g
	  	g4 r8 g16 g
	  	a8 e8 e16 d16 c16 d16
	  	e4 r8 e16 d16
	  	c4. c8 d4 e16 d c8
	  	c2 
	  	r4 e16 f8.
	  	g16 g g g g f e g
	  	g4 r8 g16 g
	  	a8 e8 e16 d16 c16 d16
	  	e4 r8 e16 d16
	  	c4. c8 d4 e16 d c8
	  	c2 
	  	r8 g8 c d
	  }
	  \new Staff 
	  {
	  	\clef "bass"
	  	c,,8 g' <c e> g
	  	c, g' <c e> g 
	  	g, d' <g b> d
	  	a e' <a c> e
	  	f, c' <f a> c
	  	c g' <c e> g 
	  	g, d' <g b> d
	  	c g' <c e> g 
	  	<c e> g <c e> g
	  	c, g' <c e> g 
	  	g, d' <g b> d
	  	a e' <a c> e
	  	f, c' <f a> c
	  	c g' <c e> g
	  	g, d' <g b> d
	  	c g' <c e> g 
	  	<c e> g <c e> g
	  	f, c' <f a> c
	  	g d' <g b> d
	  	f, c' <f a> c
	  	g d' <g b> d
	  	<g b> d <g b> d
	  	c g' <c e> g 
	  	g, d' <g b> d
	  	a e' <a c> e
	  	e, b' <e g> b
	  	f c' <f a> c
	  	g d' <g b> d
	  	c g' <c e> g 
	  	<c e> g <c e> g 
	  	c, g' <c e> g 
	  	g, d' <g b> d
	  	a e' <a c> e
	  	e, b' <e g> b
	  	f c' <f a> c
	  	g d' <g b> d
	  	c g' <c e> g
	  	<c e> g <c e> g
	  }
	  >>
	}
	\layout 
	{
		\context 
		{
			\StaffGroup
			\override StaffGrouper.staff-staff-spacing.basic-distance = #8
		}
	}
	\midi 
	{
		\tempo 4 = 72
	}
}
