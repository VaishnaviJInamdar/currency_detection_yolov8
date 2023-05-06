from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO(r'model\best_with_german.pt') #model selection

#importing videos for testing
#video paths
video = r'C:\notcdrive\jeevan\zummit\third_eye\prj-third-eye\currency_identification\data\all_test_videos.mp4'
video2 = r'C:\Users\klaus\OneDrive\Desktop\VID20230316131045.mp4'

# Export the model
#model.export(format='onnx')
url = 'https://1fc3-47-31-158-93.in.ngrok.io' 

results = model.predict(source=video2, show=True)  #update the results if source is video2
print(results)  #display results
