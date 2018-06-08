from . import PyScoreDraft

def ObjectToId(obj):
	'''
	Utility only used intenally. User don't use it.
	'''
	if type(obj) is list:
		return [ObjectToId(sub_obj) for sub_obj in obj]
	else:
		return obj.id

# generate dynamic code
g_generated_code_and_summary=PyScoreDraft.GenerateCode()
exec(g_generated_code_and_summary[0])

