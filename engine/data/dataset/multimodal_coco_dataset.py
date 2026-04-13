"""
Multimodal COCO dataset for RGB + NPY inputs.    
"""     
    
import os
from pathlib import Path
   
import numpy as np  
import torch   
import torch.nn.functional as F

from ...core import register  
from .coco_dataset import CocoDetection

 
@register()  
class MultimodalCocoDetection(CocoDetection):    
    __inject__ = ["transforms"]     
    __share__ = ["remap_mscoco_category", "cache_imgsz", "ram_cache", "num_classes"]

    def __init__(  
        self,
        img_folder,     
        npy_folder,
        ann_file,
        transforms, 
        return_masks=False,  
        remap_mscoco_category=False,
        cache_imgsz=None, 
        ram_cache=False,
        num_classes=None,
        npy_size_mismatch="error",  
        npy_dtype="float32", 
    ):
        super().__init__(     
            img_folder=img_folder,   
            ann_file=ann_file,     
            transforms=transforms,    
            return_masks=return_masks,  
            remap_mscoco_category=remap_mscoco_category,
            cache_imgsz=cache_imgsz,  
            ram_cache=ram_cache,  
            num_classes=num_classes,
        )
        self.npy_folder = str(npy_folder)   
        self.npy_size_mismatch = npy_size_mismatch
        self.npy_dtype = npy_dtype    
        self._npy_cache = {}   
 
        if self.npy_size_mismatch not in {"error", "resize"}:  
            raise ValueError("npy_size_mismatch must be one of {'error', 'resize'}")

    def __getitem__(self, idx):
        sample, target = self.load_item(idx)
        if self._transforms is not None:
            sample, target, _ = self._transforms(sample, target, self)  
        return sample, target   
  
    def load_item(self, idx):
        rgb, target = super().load_item(idx)
        image_id = self.ids[idx]
        npy = self._load_npy(image_id, rgb)     
        sample = {"rgb": rgb, "npy": npy}   
        return sample, target     
     
    def _load_npy(self, image_id, rgb_image): 
        if self.ram_cache and image_id in self._npy_cache:    
            npy_tensor = self._npy_cache[image_id]   
        else:
            npy_path = self._resolve_npy_path(image_id)
            raw = np.load(npy_path, allow_pickle=True)
            if type(raw) is np.lib.npyio.NpzFile: 
                try:    
                    npy = raw["arr_0"]     
                except KeyError as e:
                    raise ValueError(    
                        f"NPZ file must contain key 'arr_0', got keys={list(raw.keys())} for {npy_path}"
                    ) from e
                finally:
                    raw.close()
            else:     
                npy = raw

            if npy.ndim != 2:
                raise ValueError(f"NPY modality must be 2D (H, W), got shape {tuple(npy.shape)} for {npy_path}")  

            if self.npy_dtype == "float32":
                npy = npy.astype(np.float32, copy=False)   
            npy_tensor = torch.from_numpy(npy).unsqueeze(0) 
  
            if self.ram_cache:   
                self._npy_cache[image_id] = npy_tensor   
 
        rgb_w, rgb_h = rgb_image.size 
        npy_h, npy_w = npy_tensor.shape[-2:] 
        if (npy_h, npy_w) != (rgb_h, rgb_w):
            if self.npy_size_mismatch == "error":
                raise ValueError(
                    f"NPY/RGB size mismatch for image_id={image_id}: npy=({npy_h}, {npy_w}), rgb=({rgb_h}, {rgb_w})"    
                )
     
            npy_tensor = F.interpolate(    
                npy_tensor.unsqueeze(0),
                size=(rgb_h, rgb_w),  
                mode="bilinear",
                align_corners=False,    
            ).squeeze(0)     
   
        return npy_tensor  
  
    def _resolve_npy_path(self, image_id):  
        image_info = self.coco.loadImgs(image_id)[0]
        stem = Path(image_info["file_name"]).stem   
        npy_path = os.path.join(self.npy_folder, f"{stem}.npy")  
        npz_path = os.path.join(self.npy_folder, f"{stem}.npz")

        if os.path.exists(npy_path):
            return npy_path
        if os.path.exists(npz_path):    
            return npz_path     
  
        raise FileNotFoundError(    
            f"Missing modality file for image_id={image_id}. Tried: {npy_path} and {npz_path}"
        )
 
    def extra_repr(self) -> str:     
        s = super().extra_repr()
        s += f"\n npy_folder: {self.npy_folder}"
        s += f"\n npy_size_mismatch: {self.npy_size_mismatch}"    
        return s  
