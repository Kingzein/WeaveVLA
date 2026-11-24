import hashlib
import json
import random
import time
from pathlib import Path

class StreamingPool:
    """Add signed-off episodes only between visits; never mutate an active visit."""
    def __init__(self, audit, rank, world, slots, version, state=None):
        self.audit=Path(audit)
        self.rank,self.world,self.slots,self.version=rank,world,slots,version
        self.records={}
        self.pools=[[] for _ in range(slots)]
        self.state=state if state is not None else [dict(active=None,index=0,visits={},visit=0) for _ in range(slots)]
        self.refresh()
        for i,s in enumerate(self.state):
            if s['active'] is None:self.choose(i)
            elif s['active'] not in self.pools[i]:raise ValueError('Missing audited cursor')
    def refresh(self):
        for p in self.audit.glob('episode_*.json'):
            ep=int(p.stem.split('_')[-1])
            if ep%self.world!=self.rank or ep in self.records:continue
            rec=json.loads(p.read_text())
            assert rec['episode']==ep and rec['audit_version']==self.version and Path(rec['files'][-1]).parent.name==self.audit.parent.name
            rec['key']=f'calvin_d_{ep:06d}'
            self.records[ep]=rec
            self.pools[(ep//self.world)%self.slots].append(ep)
        self.last_refresh=time.monotonic()
    def choose(self,i):
        s=self.state[i];rng=random.Random(f"CALVIN_D:42:{self.rank}:{i}:{s['visit']}")
        pool=self.pools[i]; low=min(s['visits'].get(k,0) for k in pool)
        ep=rng.choice(sorted(k for k in pool if s['visits'].get(k,0)==low))
        s.update(active=ep,index=0,offset=rng.randrange(min(8,self.records[ep]['length'])),visit=s['visit']+1)
        s['visits'][ep]=low+1
    def peek(self):
        return [(dict(self.records[s['active']],offset=s['offset']),s['index']) for s in self.state]
    def advance(self):
        if time.monotonic()-self.last_refresh>=30:self.refresh()
        finished=[]
        for i,s in enumerate(self.state):
            rec=self.records[s['active']];s['index']+=1
            if s['offset']+8*s['index']>=rec['length']:
                finished.append(rec['key']);self.choose(i)
        return finished
