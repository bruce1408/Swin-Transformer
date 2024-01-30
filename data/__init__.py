from .build import build_loader as _build_loader
from .build import build_loader_single_gpu as _build_loader_single_gpu
from .data_simmim_pt import build_loader_simmim
from .data_simmim_ft import build_loader_finetune


def build_loader(config, dist=False, simmim=False, is_pretrain=False):
    if not simmim:
        if dist:
            return _build_loader(config)
        else: 
            return _build_loader_single_gpu(config)
    if is_pretrain:
        return build_loader_simmim(config)
    else:
        return build_loader_finetune(config)
