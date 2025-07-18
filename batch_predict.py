import pandas as pd
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import json

API_URL = "http://localhost:8080/predict"
INPUT_CSV_PATH = "test.csv"
OUTPUT_CSV_PATH = "result.csv"

#用API進行辨識
def predict_via_api(img_bytes_for_api):
    try:
        files = {"file":img_bytes_for_api}
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            result = json.loads(response.text) #result.text字串-->字典
            return result
        else:
            print("請求失敗", response.status_code)
            return {"predicted_class":"None", "success":False}
    except Exception as e:
        print(f"網路異常或請求API失敗: {str(e)}")
        return {"predicted_class":"None", "success":False}

#矩陣-->bytes資料
def array_transform_to_bytes(img):  
    img_byte = BytesIO() #建立一個空的BytesIO物件
    img_pil = img.astype(np.uint8)
    img_pil = Image.fromarray(img_pil, "L") #將矩陣img_pil存進img_byte_arr，變成在RAM中的PNG格式的bytes資料
    img_pil.save(img_byte, format="PNG")
    img_bytes_for_api = img_byte.getvalue() # 獲取bytes資料
    return img_bytes_for_api



#pixel-->矩陣
test_data = pd.read_csv(INPUT_CSV_PATH).values.astype(np.float32)
test_data = test_data.reshape(-1, 1, 28, 28)

#建立空矩陣存放辨識結果
csv_content = np.empty((test_data.shape[0], 1), np.uint8)  

#跑所有辨識
for i in range(test_data.shape[0]):
    img = test_data[i][0]
    img_bytes_for_api = array_transform_to_bytes(img)
    result = predict_via_api(img_bytes_for_api)        
    csv_content[i] = result["predicted_class"]

csv_content = pd.DataFrame(csv_content)
csv_content.to_csv(OUTPUT_CSV_PATH, index=False, header=False)

     
    
    
    
    
    
    

