import torch
from PIL import Image
from torchvision import transforms
from basicsr.archs.uformer_arch import Uformer  # Make sure this matches your Uformer import

# --------- SETTINGS ---------
model_path = r'C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\trained model\trained_model.pth'
input_image_path = r'"C:\Flare Removal\dataset\Flare7Kpp\test_data\real\input\input_000004.png"'  # Replace with your test image
output_image_path = r'C:\Users\sweet\Downloads\output_result.jpg'


img_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🛠️ Create model correctly
model = Uformer(img_size=img_size, embed_dim=32, in_chans=4, out_chans=6)
model.to(device)

# 🛠️ Load checkpoint
model_path = r'C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\trained model\trained_model.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)

model.eval()
print("Model loaded successfully!")

# --------- LOAD INPUT IMAGE ---------
img = Image.open(input_image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

input_tensor = transform(img).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)

# --------- INFERENCE ---------
with torch.no_grad():
    output = model(input_tensor)

# --------- SAVE OUTPUT IMAGE ---------
output_img = output.squeeze(0).cpu().clamp(0, 1)  # Remove batch dimension and clamp to [0,1]
output_img_pil = transforms.ToPILImage()(output_img)
output_img_pil.save(output_image_path)

print(f"🎯 Inference done! Output saved at: {output_image_path}")
