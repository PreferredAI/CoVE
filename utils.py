import io
import pickle

import torch


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name, device="cpu"):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)


...
# # contents = pickle.load(f) becomes...
# contents = CPU_Unpickler(f).load()
