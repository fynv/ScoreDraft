from .Instrument import Instrument
from .PyScoreDraft import InitializeInternalInstrument

PureSin_Type = 0
Square_Type = 1
Triangle_Type = 2
Sawtooth_Type = 3
NaivePiano_Type = 4
BottleBlow_Type = 5

class PureSin(Instrument):
	def __init__(self):
		self.id = InitializeInternalInstrument(PureSin_Type)

class Square(Instrument):
	def __init__(self):
		self.id = InitializeInternalInstrument(Square_Type)

class Triangle(Instrument):
	def __init__(self):
		self.id = InitializeInternalInstrument(Triangle_Type)

class Sawtooth(Instrument):
	def __init__(self):
		self.id = InitializeInternalInstrument(Sawtooth_Type)

class NaivePiano(Instrument):
	def __init__(self):
		self.id = InitializeInternalInstrument(NaivePiano_Type)

class BottleBlow(Instrument):
	def __init__(self):
		self.id = InitializeInternalInstrument(BottleBlow_Type)

