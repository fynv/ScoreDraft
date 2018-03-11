#!/usr/bin/python3

import sys
sys.path+=['../']

import ScoreDraft
from ScoreDraftNotes import *
import Meteor

doc=Meteor.Document()

seq=[do(),do(),so(),so(),la(),la(),so(5,96)]

doc.playNoteSeq(seq, ScoreDraft.NaivePiano())
doc.meteor()
doc.mixDown('HelloMeteor.wav')
