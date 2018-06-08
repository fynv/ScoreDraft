import json

Catalog= {
'Engines' : [],
'Instruments': [],
'Percussions': [],
'Singers': [] 	
}

def PrintCatalog():
	print (json.dumps(Catalog, indent=2))
