from .musicxml import ScorePartwise
from xsdata.formats.dataclass.parsers import XmlParser
try:
    from .Meteor import Document
except:
    from .Document import Document
import ly.musicxml

def _find_tempo(score):
    for part in score.part:
        for measure in part.measure:
            direction = measure.direction
            if len(direction)>0:
                sound = direction[0].sound
                if not sound is None and not sound.tempo is None:
                    return int(sound.tempo)
    return 120
    

_stepIdxs = {
    'C': 0.0,
    'D': 2.0,
    'E': 4.0,
    'F': 5.0,
    'G': 7.0, 
    'A': 9.0, 
    'B': 11.0,     
}
    
def _part_to_seq(part):
    attrtib = part.measure[0].attributes[0]
    divisions = int(attrtib.divisions)
    if 48 % divisions != 0:
        print('ScoreDraft cannot handle divisions: %d' % divisions)
        return []
    seq = []
    duration = 0
    for measure in part.measure:
        for note in measure.note:
            if duration>0 and len(note.chord)>0:
                seq += [(-1.0, -duration)]
            freq = -1.0
            if len(note.rest)<1 and len(note.pitch)>0:
                pitch = note.pitch[0]
                octave = float(pitch.octave)
                step_idx = _stepIdxs[pitch.step.name]
                if not pitch.alter is None:
                    step_idx += float(pitch.alter)
                step_idx += (octave - 4.0)*12.0
                freq = 2.0**(step_idx/12.0)            
            duration = int(note.duration[0] * 48 / divisions)            
            seq += [(freq, duration)]
    return seq    

class MusicXMLDocument(Document):
    def __init__(self, str_xml):
        Document.__init__(self)
        parser = XmlParser()
        self.score = parser.from_string(str_xml, ScorePartwise)
        self.tempo = _find_tempo(self.score)
        
    def playXML(self, instruments):
        for i in range(len(self.score.part)):
            j = i
            if j >= len(instruments):
                j = len(instruments) - 1
            part = self.score.part[i]
            seq = _part_to_seq(part)
            self.playNoteSeq(seq, instruments[j])
        
        
def from_music_xml(filename):
    with open(filename, 'r') as file:
        return MusicXMLDocument(file.read())
        
def from_lilypond(filename):
    with open(filename, 'r') as file:
        e = ly.musicxml.writer()
        e.parse_text(file.read())
        xml = e.musicxml()
        return MusicXMLDocument(xml.tostring().decode('utf-8'))
