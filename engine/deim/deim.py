"""   
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.   
"""     

import torch.nn as nn   
from ..core import register

   
__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module): 
    __inject__ = ['backbone', 'encoder', 'decoder', ]  
    
    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module,
        decoder: nn.Module,  
    ):    
        super().__init__()  
        self.backbone = backbone     
        self.decoder = decoder
        self.encoder = encoder
    
    def forward(self, x, targets=None, return_feature: bool = False, feature_level: int = 0):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        out = self.decoder(feats, targets)
        if return_feature:
            try:
                lvl = int(feature_level)
            except Exception:
                lvl = 0
            if isinstance(feats, (list, tuple)) and len(feats) > 0:
                lvl = max(0, min(lvl, len(feats) - 1))
                out = dict(out)
                out["pseudo_feat"] = feats[lvl]
        return out
  
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):    
                m.convert_to_deploy()
        return self  
