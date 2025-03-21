import os
import shutil
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from IPython.display import Video, display
from flask import Flask, request, jsonify
from flask_cors import CORS

app= Flask(__name__)
CORS(app)

d = torch.device("cpu")
mobilenet = models.mobilenet_v2(pretrained=True)
model= mobilenet.to(d)
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, 2)
mobilenet.load_state_dict(torch.load("D:\Desktop\ABC\model.pth", map_location=d))

def predicting_(img):

    img= img.convert("RGB")

    transform= transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
    
    img_tens= transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out= model(img_tens)
        prob= F.softmax(out, dim= 1)
        confi= prob[0][1].item()
        pred= torch.argmax(prob, 1).item()

    lbl= 'Real' if pred == 0 else 'Fake'
    plt.imshow(img)
    plt.title(f"Prediction: {lbl}\n({confi*100:.2f}% Fake Confidence)")
    plt.axis("off")
    plt.show()
    return (lbl)

def processing_():

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file= request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_forms= ['jpg', 'jpeg', 'png']
    vid_forms= ['mp4', 'mov']

    ext= file.filename.split('.')[-1].lower()

    if ext in img_forms:
        img = Image.open(file)
        lbl = predicting_(img)
        return jsonify({"message": f"The image is likely {lbl}."})
    elif ext in vid_forms:
        cap= cv2.VideoCapture(file)
        frames= 0
        r_frames= 0
        f_frames= 0
        s_frames= []
        while True:
            ret, frame= cap.read()
            if not ret:
                break
            if frames%30 == 0:
                rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lbl= predicting_(Image.fromarray(rgb))
                if lbl == "Real":
                    r_frames+= 1
                elif lbl == "Fake":
                    f_frames+= 1
                s_frames.append((rgb, lbl))
            frames+= 1
        cap.release()
        total_frames = r_frames + f_frames
        deepfake_percentage = (f_frames / total_frames) * 100 if total_frames > 0 else 0
        response_data = {
            "total_frames_analyzed": total_frames,
            "real_frames": r_frames,
            "fake_frames": f_frames,
            "deepfake_percentage": f"{deepfake_percentage:.2f}%",
            "sample_predictions": [{"frame": i + 1, "prediction": lbl} for i, (_, lbl) in enumerate(s_frames[:10])]
        }
        return jsonify(response_data)
    
    else:
        return jsonify({"error": "File format not supported"}), 400
    
if __name__ == "__main__":
    app.run(debug= True, host= "0.0.0.0", port= 5000)