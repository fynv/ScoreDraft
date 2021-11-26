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
            
        ly_text='\\version "2.18.2"\n'
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
    ("halfopenhihat", "f,"),
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
    ("openhighhat", "f,"),
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
    ("hhho", "f,"),
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
    ("hho", "f,"),
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
        self.instruments = []
        if 'staffs' in score:
            staffs = score['staffs']
            for staff in staffs:
                is_drum = False
                if 'is_drum' in staff:
                    is_drum = staff['is_drum']
                    
                if 'instrument' in staff:
                    ldic=locals()
                    exec('inst = ' + staff['instrument'], globals(),ldic)
                    inst=ldic["inst"]
                self.instruments += [inst] 
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
        self.playXML(self.instruments)
    
