"""Text-only, conservative entity/region parser; no BDDL, labels, or scores."""
class StructuredExtractor:
 def __init__(self,nlp):self.nlp=nlp
 def __call__(self,text):
  doc=self.nlp(text);chunks=list(doc.noun_chunks);byroot={c.root.i:c for c in chunks}
  def phrase(c):
   a=c.start
   while a<c.end and (doc[a].pos_=='DET' or doc[a].lower_ in {'both','all'}):a+=1
   return doc[a:c.end].text.lower().strip()
  def chunk(t):return byroot.get(t.i)
  entities=[];regions=[];relations=[];adjuncts=[];uncertain=[];decisions=[]
  spatial={'top','front','back','left','right','middle','center','centre','side','bottom'}
  regionset=spatial|{'layer','compartment','section','shelf','surface','area','position','location'}
  for c in chunks:
   root=c.root
   if root.pos_ not in {'NOUN','PROPN'}:continue
   p=phrase(c);lemma=root.lemma_.lower();h=root.head;owner=None;owner_source=None
   if lemma in regionset:
    of=next((t for t in root.children if t.dep_=='prep' and t.lower_=='of'),None)
    if of:
     obj=next((t for t in of.children if t.dep_=='pobj'),None)
     if obj is not None and chunk(obj) is not None:owner=phrase(chunk(obj));owner_source='of complement'
    # compound ownership only for positional heads, not "heating area".
    if owner is None and lemma in spatial:
     compounds=[t for t in root.children if t.dep_=='compound' and t.pos_ in {'NOUN','PROPN'}]
     if compounds:owner=' '.join(t.text.lower() for t in sorted(compounds,key=lambda t:t.i));owner_source='noun compound'
   reason=None
   if root.dep_=='pobj' and h.lemma_ in {'with','without'} and lemma in {'care','caution','attention','precision','accuracy'}:reason='manner'
   elif root.dep_=='pobj' and h.lemma_=='for' and lemma in {'purpose','reason','organization','balance','convenience','safety'}:reason='purpose'
   elif lemma in {'attention','care'} and h.lemma_ in {'pay','take'}:reason='abstract verb expression'
   elif lemma in {'time','moment','step','variation'} and root.dep_ in {'npadvmod','pobj'}:reason='time/process'
   if reason:
    adjuncts.append(dict(text=p,role=reason));decisions.append(dict(phrase=p,kind='adjunct',reason=reason));continue
   if owner is not None:
    regions.append(dict(text=p,owner=owner,evidence=owner_source));decisions.append(dict(phrase=p,kind='region',owner=owner,reason='explicit region ownership'))
    if owner_source=='noun compound':entities.append(dict(text=owner,role='reference',evidence='explicit compound owner'))
    continue
   if lemma in spatial and root.dep_=='pobj' and h.lower_ in {'on','at','to','in','from'}:
    relations.append(dict(text=h.text+' '+p,owner=None));uncertain.append(dict(text=p,reason='position with unresolved owner'));decisions.append(dict(phrase=p,kind='relation',reason='position; unresolved owner'));continue
   action=next((a.lemma_ for a in root.ancestors if a.pos_=='VERB'),'')
   role='candidate'
   if root.dep_ in {'dobj','obj'}:role='acted_on'
   elif root.dep_=='pobj':role='destination' if h.lower_ in {'in','into','on','onto'} else 'reference'
   elif root.dep_=='conj':role='coordinated_entity'
   entities.append(dict(text=p,role=role,action=action,evidence=root.dep_));decisions.append(dict(phrase=p,kind='entity',reason='conservative entity candidate'))
  # Keep relationships separately; no forced pronoun resolution.
  for t in doc:
   if t.dep_=='prep':
    ob=[phrase(chunk(k)) for k in t.children if chunk(k) is not None]
    if ob:relations.append(dict(text=t.text,targets=ob,governor=t.head.text))
  entities=list({e['text']:e for e in entities}.values())
  return dict(instruction=text,entities=entities,regions=regions,relations=relations,adjuncts=adjuncts,uncertain=uncertain,decisions=decisions)
