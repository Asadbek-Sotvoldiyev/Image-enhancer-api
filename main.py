import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
import io
import asyncio
from queue import Queue

app = FastAPI(title="Real-ESRGAN Image Enhancement API")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'weights/RealESRGAN_x4plus.pth'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=128, tile_pad=5, pre_pad=0, half=True, device=device)

# Navbat yaratish
processing_queue = asyncio.Queue()

async def process_queue():
    while True:
        img_array, future = await processing_queue.get()
        try:
            output, _ = upsampler.enhance(img_array)
            future.set_result(output)
        except Exception as e:
            future.set_exception(e)
        finally:
            processing_queue.task_done()

# Server ishga tushganda navbatni boshlash
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Faqat tasvir fayllari qabul qilinadi")

    try:
        img_data = await file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_array = np.array(img)

        # Navbatga qo‘shish
        future = asyncio.Future()
        await processing_queue.put((img_array, future))
        output = await future  # Natijani kutish

        output_img = Image.fromarray(output)
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=enhanced_output.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tasvirni qayta ishlashda xato: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Real-ESRGAN Image Enhancement API"}

if __name__ == "__main__":
    import uvicorn
    print(f"Device: {device}")
    uvicorn.run(app, host="0.0.0.0", port=8000)