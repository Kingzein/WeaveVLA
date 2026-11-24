"""Fail-closed contracts shared by cache, training and deployment."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROBOT_MASK_VERSION = 'visible_scene_v4_two_fingers_20261020'

def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode()); h.update(str(array.dtype).encode())
    h.update(array.tobytes())
    return h.hexdigest()

def validate_masks(rgb, masks, instances, provenance, robot, gripper, step, rgb_instruction_sha256):
    """Array integrity is necessary, but only synchronized scene evidence certifies visibility.

    RGB-predicted corrections and approximate HDF5 replays require a separately
    evaluated recipe. They cannot enter through this simulator-GT contract.
    """
    if robot is None or gripper is None:
        raise ValueError('Formal cache requires stored robot and gripper masks')
    robot, gripper, masks = map(np.asarray, (robot, gripper, masks))
    shape = np.asarray(rgb).shape[:2]
    if robot.dtype != bool or gripper.dtype != bool or masks.dtype != bool:
        raise ValueError('Mask arrays must be boolean')
    if robot.shape != shape or gripper.shape != shape or masks.shape != (len(instances), *shape):
        raise ValueError('Mask/image grid mismatch')
    labels = [x['label'] for x in instances]
    if labels.count('robot gripper') != 1 or 'robot' in labels:
        raise ValueError('Exactly one gripper and no whole-arm graph node required')
    index = labels.index('robot gripper')
    if np.any(gripper & ~robot) or not np.array_equal(masks[index], gripper):
        raise ValueError('Gripper must be contained in robot and equal its graph mask')
    objects = np.delete(masks, index, axis=0)
    if np.any(objects & robot):
        raise ValueError('Object masks overlap visible robot')
    if provenance.get('robot_mask_version') != ROBOT_MASK_VERSION:
        raise ValueError('Unverified robot mask implementation')
    if provenance.get('gripper_scope') != 'two_finger_subtrees_only':
        raise ValueError('Two-finger mask evidence required; whole gripper is incompatible')
    if provenance.get('segmentation_msaa_samples') != 0:
        raise ValueError('Segmentation must use a zero-MSAA ID framebuffer')
    for key in ('verified_rgb_instances', 'rgb_state_verified', 'scene_occlusion'):
        if provenance.get(key) is not True:
            raise ValueError(f'Missing synchronized visibility evidence: {key}')
    if not isinstance(provenance.get('scene_sha256'), str) or len(provenance['scene_sha256']) != 64:
        raise ValueError('Missing RGB scene snapshot identity')
    if provenance.get('rgb_sha256') != array_digest(rgb):
        raise ValueError('Mask RGB snapshot digest mismatch')
    if provenance.get('control_step') != step or provenance.get('rgb_instruction_sha256') != rgb_instruction_sha256:
        raise ValueError('Mask provenance belongs to a different RGB/control step')
    for key, array in [('robot_mask_sha256', robot), ('gripper_mask_sha256', gripper), ('instance_masks_sha256', masks)]:
        if provenance.get(key) != array_digest(array):
            raise ValueError(f'Mask checksum mismatch: {key}')

def model_contract(recipe):
    root = Path(__file__).parent
    value = dict(recipe=recipe,
                 source={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ('graph.py', 'base_graph.py', 'geometry.py', 'runtime.py',
                                      'prompt.py', 'text_parser_worker.py', 'depth.py', 'sync_robot_mask.py', 'robot_mask.py', 'tcp.py')})
    selected=root.parent/'language'
    value['selected_text_config']={name:hashlib.sha256((selected/name).read_bytes()).hexdigest()
                                  for name in ('selected_config.json','extract.py')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def check_graph_config(config, recipe):
    for key, value in recipe['graph'].items():
        if config.get(key) != value:
            raise ValueError(f'Checkpoint graph setting differs: {key}')
    current_contract = model_contract(recipe)
    if config.get('model_contract_sha256') != current_contract:
        # Only an explicit, tested source-to-source equivalence is accepted.
        root = Path(__file__).parent
        bridge_path = root/'equivalent_graph_versions.json'
        bridge = json.loads(bridge_path.read_text()) if bridge_path.exists() else {}
        evidence_path = root/'speed_parity.json'
        accepted = (bridge.get('old_contract') == config.get('model_contract_sha256')
                    and bridge.get('new_contract') == current_contract
                    and bridge.get('new_graph_sha256') == hashlib.sha256((root/'graph.py').read_bytes()).hexdigest()
                    and evidence_path.exists()
                    and bridge.get('evidence_sha256') == hashlib.sha256(evidence_path.read_bytes()).hexdigest())
        if not accepted:
            raise ValueError('Graph topology/implementation contract differs; do not silently load old weights')
