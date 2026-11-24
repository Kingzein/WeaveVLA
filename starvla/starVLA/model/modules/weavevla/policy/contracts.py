import hashlib
from pathlib import Path
from starVLA.model.modules.weavevla.geometry.contracts import array_digest, validate_masks
from starVLA.model.modules.weavevla.geometry.contracts import check_graph_config as check_base
from .visual_patch import MODE

def pooling_contract():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in ('visual_patch.py','masks.py','framework.py','geometry.py','runtime.py'))).hexdigest()

def check_graph_config(config,recipe):
    check_base(config,recipe)
    if config.get('visual_pooling')!=MODE or config.get('visual_pooling_contract')!=pooling_contract():
        raise ValueError('Mask pooling configuration or source contract differs')
