import re
from geometry_filter import classify_masks

# Structural background categories; task objects such as cabinet/drawer stay.
BACKGROUND_LABELS=frozenset({
 'wall','walls','floor','floors','table','tables','tabletop','table top',
 'ceiling','ceilings','ground','background','countertop','counter top','worktop',
 'wall surface','floor surface','table surface','floor tile','floor tiles',
})

def background_keyword(label):
    text=' '.join(re.findall(r'[a-z0-9]+',str(label).lower()))
    # Match complete category suffixes, preserving objects like table lamp,
    # floor mat, wall clock, tablecloth and wall-mounted cabinet.
    for term in sorted(BACKGROUND_LABELS,key=lambda x:(-len(x),x)):
        if text==term or text.endswith(' '+term):return term
    return None

def filter_masks(masks,objects,depth):
    decisions=[None]*len(objects);foreground=[]
    for i,obj in enumerate(objects):
        keyword=background_keyword(obj['label'])
        if keyword:
            decisions[i]=dict(region=i+1,status='delete',reason='background_keyword_forced_exclusion',keyword=keyword,label=obj['label'],area=int(masks[i].sum()))
        else:foreground.append(i)
    # Background categories never depend on confidence, area or plane fit.
    if foreground:
        scores=[objects[i]['sam_predicted_iou'] if 'sam_predicted_iou' in objects[i] else objects[i].get('yolo_score',objects[i].get('score',0)) for i in foreground]
        classified,_=classify_masks(depth,masks[foreground],scores)
        for i,decision in zip(foreground,classified):
            decisions[i]=dict(decision,region=i+1)
    return decisions
