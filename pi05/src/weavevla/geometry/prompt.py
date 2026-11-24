"""Selected structured parser + MiniLM forced top1, without rewriting prompts."""
import os
import json
from pathlib import Path

import numpy as np


class PromptMatcher:
    def __init__(self, root):
        import torch
        from transformers import AutoModel, AutoTokenizer
        root = Path(root)
        selected = Path(__file__).resolve().parents[1] / 'language'
        config = json.loads((selected/'selected_config.json').read_text())
        # Keep pinned spaCy binaries isolated from the policy's Python environment.
        # The same worker is used offline and online; only instructions cross it.
        import subprocess, atexit
        self.parser_process = subprocess.Popen([os.environ.get('WEAVEVLA_SPACY_PYTHON', str(root/'spacy/bin/python')),'-u',
            str(Path(__file__).with_name('text_parser_worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            text=True,bufsize=1)
        atexit.register(self.close)
        self.parsed = {}
        path = Path(os.environ.get('WEAVEVLA_MINILM', str(root/'minilm'))).resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).float().eval()
        self.config, self.vectors, self.torch = config, {}, torch

    def close(self):
        proc=getattr(self,'parser_process',None)
        if proc is not None and proc.poll() is None:
            proc.stdin.close()
            try:proc.wait(timeout=5)
            except __import__('subprocess').TimeoutExpired:proc.terminate();proc.wait(timeout=5)

    def parser(self, instruction):
        if instruction not in self.parsed:
            p=self.parser_process
            p.stdin.write(json.dumps(dict(instruction=instruction))+'\n');p.stdin.flush()
            line=p.stdout.readline()
            if not line:raise RuntimeError('Pinned text parser exited; inspect stderr')
            result=json.loads(line)
            if not result['ok']:raise RuntimeError(result['error'])
            self.parsed[instruction]=result['result']
        return self.parsed[instruction]

    def encode(self, texts):
        missing = sorted(set(texts)-self.vectors.keys())
        if missing:
            with self.torch.inference_mode():
                x = self.tokenizer(missing, padding=True, truncation=True, max_length=256, return_tensors='pt')
                y = self.model(**x).last_hidden_state
                mask = x['attention_mask'].unsqueeze(-1)
                vectors = self.torch.nn.functional.normalize((y*mask).sum(1)/mask.sum(1), dim=1).cpu().numpy()
            self.vectors.update(zip(missing, vectors))
        return np.stack([self.vectors[t] for t in texts])

    def __call__(self, instruction, objects):
        parsed = self.parser(instruction)
        names = sorted({o['label'] for o in objects if o['label'] != 'robot gripper' and len(o['points']) >= 16
                        and o.get('geometry_source', 'fresh') in ('fresh', 'static_supported')})
        matches = []
        if names and parsed['entities']:
            scores = self.encode([e['text'] for e in parsed['entities']]) @ self.encode(names).T
            for entity, row in zip(parsed['entities'], scores):
                label = names[int(row.argmax())]
                matches.append(dict(phrase=entity['text'], label=label, score=float(row.max()),
                    ids=[str(o['id']) for o in objects if o['label'] == label and len(o['points']) >= 16]))
        return dict(instruction=instruction, parsed=parsed, matches=matches,
                    prompt_ids=sorted({oid for m in matches for oid in m['ids']}))
