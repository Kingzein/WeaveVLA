"""Standard LIBERO: 40 tasks, 50 fixed initial states per task."""
import json
from pathlib import Path

SUITES = {'libero_spatial': 220, 'libero_object': 280,
          'libero_goal': 300, 'libero_10': 520}
TRIALS = 50
SETTLING_STEPS = 10

def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)

def assignments(lanes=12):
    result = {str(i): {s: [] for s in SUITES} for i in range(lanes)}
    index = 0
    for suite in SUITES:
        for task in range(10):
            for trial in range(TRIALS):
                result[str(index % lanes)][suite].append([task, trial])
                index += 1
    expected = {(s, t, r) for s in SUITES for t in range(10) for r in range(TRIALS)}
    actual = [(s, t, r) for a in result.values() for s, pairs in a.items() for t, r in pairs]
    assert len(actual) == len(set(actual)) == 2000 and set(actual) == expected
    return result

def validate_episode(root, row, expected):
    import numpy as np
    pair = (int(row['task']), int(row['trial']))
    assert pair in expected
    folder = Path(root) / f'task_{pair[0]:02d}' / f'trial_{pair[1]:02d}'
    assert json.loads((folder / 'result.json').read_text()) == row
    video = Path(root) / row['video']
    assert video.parent == folder and video.stat().st_size > 0
    a = np.load(folder / 'actions.npy', mmap_mode='r', allow_pickle=False)
    assert a.shape == (row['steps'], 7) and np.isfinite(a).all()
    assert set(np.unique(a[:, 6])).issubset({-1., 1.})
    return pair
