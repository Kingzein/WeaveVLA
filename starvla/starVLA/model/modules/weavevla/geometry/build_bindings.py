"""Regenerate prompt-to-instance bindings after mask/registry replacement."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from .prompt import PromptMatcher
from .contracts import validate_masks
from .cache import atomic_json,rgb_instruction_digest
from .geometry import unproject

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',required=True)
    p.add_argument('--mask-root',required=True);p.add_argument('--raw-depth-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--prompt-root',type=Path,required=True);a=p.parse_args()
    matcher=PromptMatcher(a.prompt_root);rows={}
    for record in json.loads(Path(a.source_manifest).read_text())['episodes']:
        f=Path(record['frames_by_offset']['0'][0])
        with np.load(f) as z:meta=json.loads(str(z['metadata']))
        rgb=np.array(Image.open(f.parent/meta['images'][0]).convert('RGB'))
        if meta['lang']!=record['instruction'] or meta['episode_key']!=record['key'] or meta['step']!=0:
            raise ValueError('Source episode identity mismatch')
        with np.load(Path(a.mask_root)/record['key']/'000000.npz') as z:
            sm=json.loads(str(z['metadata']));masks=z['masks'].copy()
            ids=[dict(id=str(i),label=str(n)) for i,n in zip(z['object_ids'],z['labels'])]
            validate_masks(rgb,masks,ids,sm['mask_provenance'],z['robot_mask'],z['gripper_mask'],0,
                           rgb_instruction_digest(rgb,record['instruction']))
        raw_path=Path(a.raw_depth_root)/'episodes'/record['key']/'000000.npz'
        with np.load(raw_path) as z:
            rm=json.loads(str(z['metadata']))
            if rm['rgb_instruction_sha256']!=rgb_instruction_digest(rgb,record['instruction']):
                raise ValueError('DA3 RGB identity mismatch')
            objects=[dict(o,points=unproject(z['depth_raw_da3_fp32'],m,z['K']),geometry_source='fresh') for o,m in zip(ids,masks)]
        rows[record['key']]=matcher(record['instruction'],objects)
        print('BINDING',len(rows),record['key'],flush=True)
    atomic_json(a.output,dict(complete=True,episodes=rows))

if __name__=='__main__':main()
