import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np

def enhancer(input_path: str, output_path: str = 'enhanced_output.png', scale: int = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'weights/RealESRGAN_x4plus.pth'

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=scale)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=256, 
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    img = Image.open(input_path).convert("RGB")
    img = np.array(img)

    output, _ = upsampler.enhance(img)

    Image.fromarray(output).save(output_path)
    return output_path

