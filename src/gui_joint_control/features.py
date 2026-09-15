"""Trusted-side regional DINOv2 preprocessing with explicit model bindings."""
from pathlib import Path
import numpy as np
from PIL import Image


def regional_rgb(image):
    image=image.convert('RGB')
    width,height=image.size
    if width < 5 or height < 5:
        raise ValueError("Screen must have at least five pixels per dimension")
    patches=[]
    for row in range(5):
        for col in range(5):
            patch=image.crop((col*width//5,row*height//5,(col+1)*width//5,(row+1)*height//5))
            patch=patch.resize((224,224),Image.Resampling.BICUBIC)
            pixels=np.asarray(patch,dtype=np.float32)/255
            pixels=(pixels-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
            patches.append(pixels.transpose(2,0,1))
    return np.stack(patches)


def public_projection(seed=2026,encoder_dim=768):
    if encoder_dim < 256:
        raise ValueError("Encoder dimension must support 256 orthogonal columns")
    matrix=np.random.default_rng(seed).standard_normal((encoder_dim,256))
    q,r=np.linalg.qr(matrix,mode='reduced')
    q *= np.where(np.diag(r)<0,-1.0,1.0)
    return q.T.copy()


class DinoRegionalEncoder:
    def __init__(self,model,projection,device='cpu'):
        import torch
        self.model=model.to(device).eval();self.device=device
        self.projection=torch.as_tensor(projection,dtype=torch.float32,device=device)
        if self.projection.ndim != 2 or self.projection.shape[0]!=256:
            raise ValueError("Public projection must be 256 x encoder_dim")
        self.model.requires_grad_(False)

    def __call__(self,image_path):
        import torch
        with Image.open(image_path) as image:
            patches=regional_rgb(image)
        with torch.no_grad():
            output=self.model(pixel_values=torch.from_numpy(patches).to(self.device))
            cls=output.last_hidden_state[:,0,:]
            features=cls@self.projection.T
            features=features/features.norm(dim=-1,keepdim=True).clamp_min(1)
        return features.cpu().numpy().astype(np.float64)

    @classmethod
    def from_pretrained(cls,model_id,revision,projection_path,device='cpu'):
        from transformers import AutoModel
        if not Path(model_id).is_dir() and (not revision or len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower())):
            raise ValueError("Remote model requires an immutable 40-hex revision")
        projection=np.load(projection_path,allow_pickle=False)
        model=AutoModel.from_pretrained(model_id,revision=revision,trust_remote_code=False)
        return cls(model,projection,device)
