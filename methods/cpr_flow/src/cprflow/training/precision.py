from contextlib import nullcontext

import torch


class Precision:
    def __init__(self, device, mode="bf16_if_supported"):
        self.device = torch.device(device)
        self.dtype = None
        if self.device.type == "cuda" and mode != "fp32":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if mode not in ("bf16_if_supported", "fp32"):
            raise ValueError("Supported precision: fp32 or bf16_if_supported")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.dtype else nullcontext()
