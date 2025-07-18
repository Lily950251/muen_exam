這是沐恩生醫光電公司出的測驗
包含以下檔案：
1. main.py：用 Fastapi 建立的 API，在 Prompt 中輸入 uvicorn main:app --port 沒有被占用的port號，即可運行！
此 API 提供上傳包含數字 0~9 的圖片，回傳辨識內容的服務。
上傳圖片最好為 jpg 或 png 檔，會回傳 JSON 格式的辨識內容，回傳結果範例如下：
{
	"predicted_calss":5,
	"success": true
}
其中，"predicted_calss" 後的數字，即是辨識結果。

2. load_model.py：導入辨識模型的程式，須和 main.py 放在同一個資料夾。

3. model_weights.pth：模型權重，須和 load_model.py 及 main.py 放在同一個資料夾。

4. batch_predict.py：批次預測程式碼。可以讀取包含數張帶有數字的圖片像素值之 csv 檔，並藉由 API 辨識圖片中的數字，並將結果存成 result.csv 檔。
*運行前請先啟動 API。

csv 檔說明：每一個 row 為一張圖片的像素資料，限制為28 X 28 像素大小的圖片，檔案格式請參考 test.csv。

5. test_and_result.zip：包含 test.csv 及 result.csv。test.csv 為含 28000 筆測試資料的 csv 檔。result.csv 則是藉由 batch_predict.py 辨識 test.csv 產生的結果檔案。

6. make_Docker.zip：包含建立 Docker 映像檔所需要的檔案：
	Dockerfile
	load_model.py
	main.py
	model_weights.pth
	requirements.txt
在 Prompt 中移至此資料夾，並執行以下命令即可！
建立映像檔 --> docker build -t docker_image .
運行容器 --> docker run -p 8080:8000 docker_image
