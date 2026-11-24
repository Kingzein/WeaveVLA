"""Current audited D cache + official action/image packing, with contexts."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import os
from threading import Lock

import numpy as np
from PIL import Image
from starVLA.model.modules.weavevla.geometry.prompt import PromptMatcher
from starVLA.model.modules.weavevla.geometry.geometry import default_recipe, recipe_digest
from .data_contract import action_chunk, normalize_actions

W = Path(__file__).resolve().parent
match_lock = Lock()


def graph_recipe(manifest):
    recipe = default_recipe()
    recipe.update(version='weavevla_calvin_d_metric',
        calvin_cache={k:manifest[k] for k in ('geometry_recipe','track_recipe','sam_recipe','detector_recipe')})
    return recipe


@lru_cache(maxsize=1)
def matcher():
    return PromptMatcher(Path(os.environ['WEAVE_PROMPT_ROOT']))


@lru_cache(maxsize=1)
def action_statistics():
    return json.loads(Path(os.environ['WEAVE_CALVIN_ACTION_STATS']).read_text())['franka']['action']


@lru_cache(maxsize=24)
def episode(files, stamps, expected_hash):
    assert len(files) == len(stamps) == 1
    p = Path(files[0])
    st = p.stat()
    if (st.st_size, st.st_mtime_ns, st.st_ino) != stamps[0]:
        raise ValueError(('Audited D cache changed', str(p)))
    # Full SHA is checked once per episode load, not for every control sample.
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected_hash
    with np.load(p, allow_pickle=False) as z:
        assert str(z['recipe']) == p.parent.name
        return dict(maps=z['maps'], points=z['points'], frames=json.loads(str(z['metadata'])),
                    actions=z['actions'], rgb=z['rgb_static'], wrist=z['rgb_gripper'],
                    rgb_sha=z['rgb_sha256'], language=str(z['language']))


def read_frame(r, index, bindings, allow_provisional=False, data_root=W):
    if allow_provisional:
        raise ValueError('D training requires its complete current-mask audited manifest')
    data = episode(tuple(r['files']), tuple(map(tuple, r['identity'])), r['sha256'])
    t = r.get('offset', 0)+8*index
    if not 0 <= t < len(data['actions']):
        raise IndexError(t)
    rgb = data['rgb'][t]
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == data['rgb_sha'][t].decode()
    # Exactly the official PIL RGB resize. The graph's semantic map follows the
    # same coordinate transform with nearest-neighbor resampling; geometry stays metric.
    obs = [Image.fromarray(image).resize((224, 224)) for image in (rgb, data['wrist'][t])]
    maps = np.asarray(Image.fromarray(data['maps'][t]).resize((224, 224), Image.Resampling.NEAREST))
    frame = data['frames'][t]
    labels = {int(k):v for k, v in frame['labels'].items()}
    fingers = [i for i, label in labels.items() if label == 'gripper fingers']
    nodes = []
    for original in frame['nodes']:
        node = original.copy()
        a, b = node.pop('point_range')
        node['points'] = data['points'][a:b]
        node['visual_mask'] = np.isin(maps, fingers) if node['id'] == 'gripper' else maps == int(node['id'])
        nodes.append(node)
    language = data['language']
    assert language == r['language']
    with match_lock:
        prompt = matcher()(language, nodes)['prompt_ids']
    chunk = normalize_actions(action_chunk(data['actions'], t), action_statistics())
    ex = dict(image=obs, lang=language, action=chunk, pilot_objects=nodes, geometry_recipe_sha256=bindings)
    ctx = dict(episode_key=r['key'], step=t, prompt_ids=prompt, sampling_offset=t % 8, geometry_generation=0)
    return ex, ctx
