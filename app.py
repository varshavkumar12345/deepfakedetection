import streamlit as st
import cv2
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# Title
st.title("🎥 DeepFake Video Detector")
st.write("Upload a video and detect DeepFake frames using a trained MobileNetV2 model.")

# Upload Video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
model_file = "deepfake_model.pth"

if video_file is not None:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_file)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 2)
    mobilenet.load_state_dict(torch.load(model_file, map_location=device))
    mobilenet = mobilenet.to(device)
    mobilenet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def is_deepfake(frame):
        with torch.no_grad():
            img_tensor = transform(frame).unsqueeze(0).to(device)
            output = mobilenet(img_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, 1).item()
            return predicted == 1, probabilities[0][0].item(), probabilities[0][1].item()

    # Process Video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_frames = 0
    real_frames = 0
    sampled_frames = []

    with st.spinner("Analyzing video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                is_fake, real_conf, fake_conf = is_deepfake(rgb_frame)
                label = 'Fake' if is_fake else 'Real'
                sampled_frames.append((rgb_frame, label))
                if is_fake:
                    fake_frames += 1
                else:
                    real_frames += 1
            frame_count += 1
        cap.release()

    total_sampled = fake_frames + real_frames
    deepfake_percent = (fake_frames / total_sampled * 100) if total_sampled > 0 else 0

    # Show Results
    st.subheader("📊 Summary")
    st.write(f"🔍 Total frames analyzed: {total_sampled}")
    st.write(f"🟢 Real frames: {real_frames}")
    st.write(f"🔴 Fake frames: {fake_frames}")
    st.write(f"📈 DeepFake Percentage: {deepfake_percent:.2f}%")

    # Show Sampled Frames
    st.subheader("🖼️ Sample Frames")
    num_show = min(5, len(sampled_frames))
    cols = st.columns(num_show)
    for i in range(num_show):
        img, label = sampled_frames[i]
        cols[i].image(img, caption=f'{label}', use_column_width=True)

    os.unlink(video_path)
