from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from model import CNN

app = FastAPI()

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)


    return {
        "label": classes [pred.item()],
        "confidence": float(conf.item())
    }

