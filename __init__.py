# __init__.py

from .uso_nodes import USOLoader, USOSampler

NODE_CLASS_MAPPINGS = {
    "USOLoader": USOLoader,
    "USOSampler": USOSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "USOLoader": "USO Model Loader (flux-fp8)",
    "USOSampler": "USO Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]