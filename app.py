from flask import Flask, render_template, request ,url_for
from PIL import Image
import torch
import os
import torchvision.models as models
from torchvision.transforms import transforms
from io import BytesIO
import base64

app = Flask(__name__, template_folder='templates', static_folder='static')

model = models.efficientnet_b3(weights=None)  

model.classifier[1] = torch.nn.Linear(in_features=1536, out_features=120)
model.load_state_dict(torch.load("effnetb3.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with open("breeds.txt", 'r') as f:
    data = [line.strip() for line in f]

def preprocess_img(img):
    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transforms_test(img).unsqueeze(0).to(device)
    return img

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")

    preprocessedImg = preprocess_img(img)

    with torch.no_grad():
        output = model(preprocessedImg)
        predicted_class = torch.argmax(output, dim=1).item()

    img_buffer = BytesIO()
    img.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")


    return render_template("prediction.html",
                            prediction_text=f"The breed is {data[predicted_class]}",
                            img_data = img_base64)

if __name__ == "__main__":
    app.run(debug=True)