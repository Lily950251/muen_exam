from fastapi import FastAPI
from fastapi import File, UploadFile 
import load_model
import torch
import torchvision.transforms as transforms 
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse  

model_loaded_successfully = True

try:
    model = load_model.CNNTransformer()  #載入模型
    model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
    model.eval()

except Exception as e:
    model_loaded_successfully = False
    model_loaded_error = JSONResponse(status_code = 500, content = {'success': False, 'error': f"模型載入失敗: {str(e)}"})

# 定義轉換
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),            # → [1, 28, 28] 且自動除以 255
])

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"歡迎使用此API辨識圖片中的數字"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model_loaded_successfully == True:
        try:
            img_bytes = await file.read()  # 載入圖片
            img = Image.open(BytesIO(img_bytes)).convert("L")  # 轉為灰階
            img_tensor = transform(img).unsqueeze(0)  # → [1, 1, 28, 28] 增加 batch 維度
            with torch.no_grad():
                output = model(img_tensor)  # → [1, 10]
                pred = torch.argmax(output, dim=1)
                return JSONResponse(content={"predicted_class": pred.item(), "success": True})
        
        except Exception as e:
            return JSONResponse(status_code=500, content={'success': False, 'error': f"辨識失敗: {str(e)}"})

    else:
        return model_loaded_error


