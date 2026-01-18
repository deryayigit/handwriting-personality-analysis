import torch
import torch.nn.functional as F
import timm
from .model import ViTClassifier
from PIL import Image
from torchvision import transforms
import math

#eğitilmiş vitin kullanımı
def load_model(model_path: str):
    ckpt = torch.load(model_path, map_location="cpu")

    model = ViTClassifier(
        model_name=ckpt["model_name"],
        num_classes=len(ckpt["class_names"]),
        pretrained=False
    )

    #ağırlıkları yükle, değerlendir ve çıktı oluştur
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return model, ckpt["class_names"], ckpt["img_size"]

#el yazısındaan kişilik tahmini aşaması için entropy ile kararsızlık ölçtük sonra predict_image kişilk tahmini yaptık

def compute_entropy(probs):
    return -sum(p * math.log(p + 1e-8) for p in probs)


def predict_image(model_path: str, image_path: str):
    model, class_names, img_size = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].tolist()

    #Trait
    trait_scores = [p * 100 for p in probs]

    #Model Skoru
    model_conf = max(trait_scores)

    #Entropy skoru
    entropy = compute_entropy(probs)
    max_entropy = math.log(len(probs))
    entropy_conf = (1 - entropy / max_entropy) * 100

    return {
        "traits": trait_scores,
        "model_confidence": model_conf,
        "entropy_confidence": entropy_conf
    }
