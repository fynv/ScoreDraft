from .Instrument import Instrument
from .PyScoreDraft import CreateInternalInstrument, DelInternalInstrument
from .Catalog import Catalog

PureSin_Type = 0
Square_Type = 1
Triangle_Type = 2
Sawtooth_Type = 3
NaivePiano_Type = 4
BottleBlow_Type = 5

class InternalInstrument(Instrument):
	def __del__(self):
		DelInternalInstrument(self.m_cptr)

Catalog['Instruments'] += ['PureSin'+' - internal']
class PureSin(InternalInstrument):
	def __init__(self):
		self.m_cptr = CreateInternalInstrument(PureSin_Type)

Catalog['Instruments'] += ['Square'+' - internal']
class Square(InternalInstrument):
	def __init__(self):
		self.m_cptr = CreateInternalInstrument(Square_Type)

Catalog['Instruments'] += ['Triangle'+' - internal']
class Triangle(InternalInstrument):
	def __init__(self):
		self.m_cptr = CreateInternalInstrument(Triangle_Type)

Catalog['Instruments'] += ['Sawtooth'+' - internal']
class Sawtooth(InternalInstrument):
	def __init__(self):
		self.m_cptr = CreateInternalInstrument(Sawtooth_Type)

Catalog['Instruments'] += ['NaivePiano'+' - internal']
class NaivePiano(InternalInstrument):
	def __init__(self):
		self.m_cptr = CreateInternalInstrument(NaivePiano_Type)

Catalog['Instruments'] += ['BottleBlow'+' - internal']
class BottleBlow(InternalInstrument):
	def __init__(self):
		self.m_cptr = CreateInternalInstrument(BottleBlow_Type)

