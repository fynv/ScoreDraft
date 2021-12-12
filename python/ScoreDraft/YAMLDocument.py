from xsdata.formats.dataclass.parsers import XmlParser
import ly.musicxml
import yaml

from .MusicXMLDocument import MusicXMLDocument
from .Initializers import *

class YAMLScore:
    def __init__(self, str_yaml):
        self.score = yaml.safe_load(str_yaml)['score']
        
    def to_ly(self):       
        tempo = 120
        if 'tempo' in self.score:
            tempo = int(self.score['tempo'])
            
        title = ""
        if 'title' in self.score:
            title = self.score['title']
            
        composer = ""
        if 'composer' in self.score:
            composer = self.score['composer']
            
        ly_text='\\version "2.18.2"\n'
        
        if title!="" or composer!="":
            ly_text+='\\header\n'
            ly_text+='{\n'
            if title!="":
                ly_text += '\ttitle = "' + title + '"\n'
            if composer!="":
                ly_text += '\tcomposer = "' + composer + '"\n'
            ly_text+='}\n'
            
        ly_text+='\\score\n'
        ly_text+='{\n'
           
        ly_text+='\t<<\n'
    
        if 'staffs' in self.score:
            staffs = self.score['staffs']
            for staff in staffs:
                is_drum = False
                if 'is_drum' in staff:
                    is_drum = staff['is_drum']
                content = staff['content']
                lines = content.splitlines()
                if is_drum:
                    ly_text+='\t\t\\drums\n'
                else:
                    relative = ""
                    if 'relative' in staff:
                        relative = ' \\relative '+staff['relative']
                    ly_text+='\t\t\\new Staff' + relative +'\n'
                ly_text+='\t\t{\n'
                for line in lines:
                    ly_text+='\t\t\t'+line+'\n'
                ly_text+='\t\t}\n'
                
        ly_text+='\t>>\n'
        ly_text+='\t\\layout {}\n'
        ly_text+='\t\\midi\n'
        ly_text+='\t{\n'
        ly_text+='\t\t\\tempo 4 = '+ str(tempo) +'\n'
        ly_text+='\t}\n'
        ly_text+='}\n'
        return ly_text
        
GM_Drum_Map = [
    ("acousticbassdrum", "b,,"),
    ("acousticsnare", "d,"),
    ("electricsnare", "e,"),
    ("halfopenhihat", "bes,"),
    ("chinesecymbal", "e"),
    ("highfloortom", "g,"),  
    ("crashcymbala", "cis"),
    ("crashcymbalb", "a"),
    ("splashcymbal", "g"),
    ("shortwhistle", "b'"),
    ("opentriangle", "a''"),
    ("mutetriangle", "aes''"),
    ("lowfloortom", "f,"),
    ("closedhihat", "fis,"),
    ("openhighhat", "bes,"),
    ("crashcymbal", "cis"),
    ("ridecymbala", "ees"),
    ("ridecymbalb", "b"), 
    ("openhibongo", "c'"),
    ("mutehibongo", "c'"),
    ("openlobongo", "cis'"),
    ("mutelobongo", "cis'"),
    ("openhiconga", "ees'"),
    ("mutehiconga", "d'"),
    ("openloconga", "e'"), 
    ("muteloconga", "e'"),
    ("hisidestick", "cis,"),
    ("losidestick", "cis,"),
    ("longwhistle", "c''"),
    ("hiwoodblock", "e''"),
    ("lowoodblock", "f''"),
    ("pedalhihat", "aes,"),
    ("ridecymbal", "ees"),  
    ("shortguiro", "cis''"),
    ("tambourine", "fis"),
    ("lowmidtom", "b,"),      
    ("hitimbale", "f'"),
    ("lotimbale", "fis'"),
    ("sidestick", "cis,"),    
    ("longguiro", "d''"),
    ("vibraslap", "bes"),
    ("opencuica", "g''"),
    ("mutecuica", "fis''"),
    ("himidtom", "c"),
    ("bassdrum", "c,"),
    ("ridebell", "f"),
    ("handclap", "ees,"),
    ("triangle", "a''"),
    ("highhat", "aes,"),
    ("hightom", "d"),
    ("lowtom", "a,"),
    ("cowbell", "aes"),
    ("hibongo", "c'"),
    ("lobongo", "cis'"),
    ("hiconga", "ees'"),
    ("loconga", "e'"),
    ("hiagogo", "g'"),
    ("loagogo", "aes'"),    
    ("maracas", "bes'"),
    ("cabasa", "a'"),
    ("tamtam", "fis"),
    ("claves", "ees''"),
    ("snare", "d,"),
    ("tomfh", "g,"),
    ("tomfl", "f,"),   
    ("tomml", "b,"),
    ("tommh", "c"),
    ("cymca", "cis"),
    ("cymcb", "a"),
    ("cymra", "ees"),
    ("cymrb", "b"),
    ("cymch", "e"),
    ("guiro", "cis''"),
    ("tomh", "d"),
    ("toml", "a,"),
    ("hhho", "bes,"),
    ("cymc", "cis"),
    ("cymr", "ees"), 
    ("cyms", "g"),   
    ("boho", "c'"),
    ("bohm", "c'"),
    ("bolo", "cis'"),
    ("bolm", "cis'"),
    ("cgho", "ees'"),
    ("cghm", "d'"),
    ("cglo", "e'"),
    ("cglm", "e'"),
    ("timh", "f'"),
    ("timl", "fis'"),
    ("guis", "cis''"),
    ("guil", "d''"),
    ("tamb", "fis"),
    ("vibs", "bes"),
    ("cuio", "g''"),
    ("cuim", "fis''"),
    ("trio", "a''"),
    ("trim", "aes''"),
    ("bda", "b,,"),
    ("sna", "d,"),
    ("sne", "e,"),
    ("hhc", "fis,"),
    ("hho", "bes,"),
    ("hhp", "aes,"),
    ("boh", "c'"),
    ("bol", "cis'"),
    ("cgh", "ees'"),
    ("cgl", "e'"),
    ("agh", "g'"),
    ("agl", "aes'"),
    ("ssh", "cis,"),
    ("ssl", "cis,"),
    ("gui", "cis''"),
    ("cab", "a'"),
    ("mar", "bes'"),
    ("whs", "b'"),
    ("whl", "c''"),
    ("wbh", "e''"),
    ("wbl", "f''"),
    ("tri", "a''"),
    ("bd", "c,"),    
    ("sn", "d,"),
    ("hh", "aes,"),
    ("rb", "f"),
    ("cb", "aes"),
    ("ss", "cis,"),
    ("hc", "ees,"),
    ("tt", "fis"),
    ("cl", "ees''")
]


_stepIdxs = {
    'C': 0.0,
    'D': 2.0,
    'E': 4.0,
    'F': 5.0,
    'G': 7.0, 
    'A': 9.0, 
    'B': 11.0,     
}

class YAMLDocument(MusicXMLDocument):
    def __init__(self, yaml_score):
        score = yaml_score.score
        tempo = 120
        if 'tempo' in score:
            tempo = int(score['tempo'])
            
        ly_text='\\version "2.18.2"\n'
        ly_text+='\\score\n'
        ly_text+='{\n'
        
        ly_text+='\t<<\n'
        
        inst = KarplusStrongInstrument()
        self.tracks = []
        if 'staffs' in score:
            staffs = score['staffs']            
            for staff in staffs:                
                is_drum = False
                if 'is_drum' in staff:
                    is_drum = staff['is_drum']
                    
                is_vocal = False
                if 'is_vocal' in staff:
                    is_vocal = staff['is_vocal']
                    
                if not is_vocal:
                    if 'instrument' in staff:
                        ldic=locals()
                        exec('inst = ' + staff['instrument'], globals(),ldic)
                        inst=ldic["inst"]
                
                else:
                    ldic=locals()
                    exec('singer = ' + staff['singer'], globals(),ldic)
                    singer=ldic["singer"]
                        
                    if 'converter' in staff:
                        exec('singer.setLyricConverter('+staff['converter']+')', globals(),locals())
                    
                    if 'CZMode' in staff and staff['CZMode']:
                        singer.setCZMode()                        
                    
                sweep = 0.0
                if 'sweep' in staff:
                    sweep = staff['sweep']
                
                content = staff['content']
                if is_drum:
                    for pair in GM_Drum_Map:
                        content = content.replace(pair[0], pair[1])                
                lines = content.splitlines()
                
                if is_drum:
                    ly_text+='\t\t\\new Staff'
                else:
                    relative = ""
                    if 'relative' in staff:
                        relative = ' \\relative '+ staff['relative']
                    ly_text+='\t\t\\new Staff' + relative +'\n'
                    
                ly_text+='\t\t{\n'
                for line in lines:
                    ly_text+='\t\t\t'+line+'\n'
                ly_text+='\t\t}\n'
                
                if is_vocal:
                    track_info = {
                        "type": "vocal",
                        "singer": singer,
                        "utau" : staff['utau']
                    }
                else:
                    track_info = {
                        "type": "regular",
                        "instrument": inst,
                        "sweep": sweep
                    }
                
                self.tracks += [track_info] 
                
                if not is_drum and 'pedal' in staff:
                    pedal = staff['pedal']
                    lines = pedal.splitlines()
                    ly_text+='\t\t\\drums\n'                   
                    ly_text+='\t\t{\n'
                    for line in lines:
                        ly_text+='\t\t\t'+line+'\n'
                    ly_text+='\t\t}\n'
                    
                    track_info = {
                        "type": "pedal"                    
                    }                    
                    self.tracks += [track_info] 
                
                
        ly_text+='\t>>\n'
        ly_text+='\t\\layout {}\n'
        ly_text+='\t\\midi\n'
        ly_text+='\t{\n'
        ly_text+='\t\t\\tempo 4 = '+ str(tempo) +'\n'
        ly_text+='\t}\n'
        ly_text+='}\n'
        
        e = ly.musicxml.writer()
        e.parse_text(ly_text)
        xml = e.musicxml()        
        MusicXMLDocument.__init__(self, xml.tostring().decode('utf-8'))
    
    def play(self):
        num_parts = len(self.score.part)
        for i in range(num_parts):
            part = self.score.part[i]
            track_info = self.tracks[i]
            if track_info['type'] == "pedal":
                continue
                
            if track_info['type'] == "vocal":
                attrtib = part.measure[0].attributes[0]
                divisions = int(attrtib.divisions)
                if 48 % divisions != 0:
                    print('ScoreDraft cannot handle divisions: %d' % divisions)
                    continue
                    
                utau = track_info['utau']              
                syllables = []
                utau = utau.replace('\n', ' ')               
                for syll in utau.split(' '):
                    text = syll
                    end_sentence = False
                    if len(text)>0 and text[-1] == '.':
                        end_sentence = True
                        text = text[0: -1]
                    if text!="":
                        syllables += [(text, end_sentence)]
                num_syll = len(syllables)
                
                seq = []
                sentence = []
                i_syll = 0
                new_syll = True
                for measure in part.measure:
                    for note in measure.note:
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
                        
                        if freq < 0.0:
                            if len(sentence)>0:
                                seq += [sentence]
                                sentence = []
                                new_syll = True
                            seq += [(freq, duration)]
                        else:
                            slur = False
                            if len(note.notations)>0:
                                if len(note.notations[0].slur)>0 and note.notations[0].slur[0].type.value!="stop":                                    
                                    slur = True
                            if new_syll:
                                sentence += [syllables[i_syll][0]]
                            sentence += [(freq, duration)]
                            
                            if slur:
                                new_syll = False
                            else:
                                new_syll = True
                                if i_syll < num_syll -1:
                                    i_syll += 1
                                if syllables[i_syll][1]:
                                    seq += [sentence]
                                    sentence = []
                if len(sentence)>0:
                    seq += [sentence]
                    sentence = []
                    
                self.sing(seq, track_info['singer'])
                
                continue
                
            sustain_ranges = []
            if i < num_parts - 1 and self.tracks[i+1]['type'] == "pedal":
                part_s = self.score.part[i + 1]
                attrtib_s = part_s.measure[0].attributes[0]
                divisions_s = int(attrtib_s.divisions)
                if 48 % divisions_s != 0:
                    print('ScoreDraft cannot handle divisions: %d' % divisions_s)
                    continue
                    
                pos = 0
                for measure in part_s.measure:
                    for note in measure.note:
                        duration = int(note.duration[0] * 48 / divisions_s)
                        if len(note.rest)<1:
                            sustain_ranges += [(pos, pos + duration)]
                        pos += duration

            # part -> seq
            attrtib = part.measure[0].attributes[0]
            divisions = int(attrtib.divisions)
            if 48 % divisions != 0:
                print('ScoreDraft cannot handle divisions: %d' % divisions)
                continue
                
            seq = []
            duration = 0
            pos = 0
            i_range = 0
            for measure in part.measure:
                for note in measure.note:
                    sustain = False
                    while i_range < len(sustain_ranges):
                        if pos >= sustain_ranges[i_range][0]:
                            if pos < sustain_ranges[i_range][1]:
                                sustain = True
                                break
                            else:
                                i_range += 1
                        else:
                            break
                    
                    freq = -1.0
                    if len(note.rest)<1 and len(note.pitch)>0:
                        pitch = note.pitch[0]
                        octave = float(pitch.octave)
                        step_idx = _stepIdxs[pitch.step.name]
                        if not pitch.alter is None:
                            step_idx += float(pitch.alter)
                        step_idx += (octave - 4.0)*12.0
                        freq = 2.0**(step_idx/12.0)            
                    
                    if duration>0 and len(note.chord)>0:
                        if track_info['sweep']>0:
                            duration = int(duration*(1.0 - track_info['sweep']))
                        seq += [(-1.0, -duration)]
                        pos -= duration
                    else:
                        duration = int(note.duration[0] * 48 / divisions)
                        
                    if sustain:
                        seq += [(freq, sustain_ranges[i_range][1] - pos)]
                        seq += [(-1.0, pos + duration -sustain_ranges[i_range][1])]
                    else:                        
                        seq += [(freq, duration)]
                    pos += duration
            
            self.playNoteSeq(seq, track_info['instrument'])
    
