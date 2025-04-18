import streamlit as st
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import json
import os

# Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Rechargement du modèle pretrained VGG16
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 10)

# 2. Chargement des poids enregistrés
model.load_state_dict(torch.load(
    "checkpoints/best_model_transfert_augment.pth",
    map_location=device
))

# 3. instanciation du mode eval
model.to(device)
model.eval()

# Chargement des labels
with open("class_to_label.json") as f:
    class_to_label = json.load(f)

with open("label_mapping.json") as f:
    label_mapping = json.load(f)

# Prétraitement
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ]
)

# Titre Streamlit
st.title("🐶 Prédicteur de race de chien")

# Folder path
folder_path = "data_10_classes"

files = []
for folder_class in os.listdir(folder_path):
	subfiles = os.listdir(os.path.join(folder_path, folder_class))
	subfiles = [os.path.join(folder_class, subfile) for subfile in subfiles]
	files.extend(subfiles)

# Dropdown menu
selected_file = st.selectbox("Select a file", files)

# Optional: Display selected file path or content
st.write(f"Selected file: {selected_file}")
# st.text(open(os.path.join(folder_path, selected_file)).read())

if selected_file is not None:
    image = Image.open(os.path.join(folder_path, selected_file)).convert("RGB")
    st.image(image, caption="Image uploadée", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        prediction = class_to_label[str(predicted.item())]

    st.success(f"Race prédite : **{prediction}**")
    st.success(f"Race réelle: **{class_to_label[str(label_mapping[selected_file.split('-')[0]])]}**")
