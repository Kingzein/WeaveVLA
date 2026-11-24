"""Pinned spaCy environment; JSON lines only on stdout, no detector inputs."""
import json,sys,importlib.util
from pathlib import Path
import spacy
root=Path(__file__).resolve().parents[1]/'language'
cfg=json.loads((root/'selected_config.json').read_text())['extraction']
assert spacy.__version__==cfg['spacy_version']
nlp=spacy.load(cfg['spacy_model'],disable=['ner'])
assert nlp.meta['version']==cfg['spacy_model_version']
spec=importlib.util.spec_from_file_location('selected_extractor',root/'extract.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
parser=module.StructuredExtractor(nlp)
for line in sys.stdin:
 try:result=dict(ok=True,result=parser(json.loads(line)['instruction']))
 except Exception as exc:result=dict(ok=False,error=repr(exc))
 print(json.dumps(result),flush=True)
