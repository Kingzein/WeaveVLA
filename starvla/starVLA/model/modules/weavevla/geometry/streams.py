"""Disjoint persistent episode slots, with exactly resumable next-frame cursors."""
import random,hashlib,copy

class Streams:
    def __init__(self, records, rank, world=4, slots=12, state=None, random_offset=False):
        self.rank=rank;self.random_offset=random_offset
        pool=sorted(records,key=lambda r:r['key'])
        random.Random(42).shuffle(pool)
        self.shard=pool[rank::world]
        self.pools=[self.shard[i::slots] for i in range(slots)]
        assert all(self.pools)
        self.state=copy.deepcopy(state) if state is not None else [dict(cycle=0,pos=0,frame=0) for _ in range(slots)]
        assert len(self.state)==slots
        self.orders=[self.order(i,s['cycle']) for i,s in enumerate(self.state)]
        for i,s in enumerate(self.state):
            # Existing checkpoint cursors/memory belong to offset0. Complete that
            # episode before drawing a new offset; never reinterpret its cursor.
            if 'offset' not in s:s['offset']=0 if state is not None else self.draw_offset(i)
            assert 0<=s['offset']<8
            if not random_offset:assert s['offset']==0,'Offset checkpoint requires random_offset mode'

    def draw_offset(self,i):
        if not self.random_offset:return 0
        s=self.state[i];r=self.orders[i][s['pos']]
        key=f"20261013:{self.rank}:{i}:{s['cycle']}:{s['pos']}:{r['key']}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%8

    def record(self,i):
        s=self.state[i];r=self.orders[i][s['pos']]
        if not self.random_offset:return r
        frames=r['frames_by_offset'][str(s['offset'])]
        assert frames,'An offset must contain at least one real observation'
        return dict(r,frames=frames,offset=s['offset'])

    def order(self,i,cycle):
        x=self.pools[i].copy();random.Random(20261013+cycle*17+i).shuffle(x);return x

    def peek(self):
        return [(self.record(i),s['frame']) for i,s in enumerate(self.state)]

    def advance(self):
        finished=[]
        for i,s in enumerate(self.state):
            r=self.record(i);s['frame']+=1
            if s['frame']==len(r['frames']):
                finished.append(r['key']);s['frame']=0;s['pos']+=1
                if s['pos']==len(self.orders[i]):
                    s['cycle']+=1;s['pos']=0;self.orders[i]=self.order(i,s['cycle'])
                s['offset']=self.draw_offset(i)
        return finished
