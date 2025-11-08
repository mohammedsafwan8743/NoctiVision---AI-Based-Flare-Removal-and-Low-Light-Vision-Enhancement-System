import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from basicsr.archs.uformer_arch import Uformer
from basicsr.dpt.models import DPTDepthModel
from basicsr.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from basicsr.utils.flare_util import predict_flare_from_6_channel

# ------------------- Setup ----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

depth_transform = Compose([
    Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet(),
])

@st.cache_resource
def load_model(model_path, output_ch=6):
    model = Uformer(img_size=512, img_ch=4, output_ch=output_ch)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model

@st.cache_resource
def load_depth_model(model_path):
    depth_model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_model.eval().to(device)
    return depth_model

# Load models
model = load_model(r"C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\final train\final_trained_model.pth")
depth_model = load_depth_model(r"C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\DPT\dpt_hybrid-midas-501f0c75.pt")

# ------------------- Preprocess ----------------------
def preprocess(img_pil):
    orig_w, orig_h = img_pil.size

    # Resize image to 512x512 for model input
    resized_img = img_pil.resize((512, 512))
    img_tensor = transforms.ToTensor()(resized_img).unsqueeze(0).to(device)

    # Resize again for depth model
    depth_input_np = np.uint8(np.array(img_pil.resize((384, 384))))
    transformed = depth_transform({"image": depth_input_np})["image"]

    with torch.no_grad():
        depth_pred = depth_model(torch.from_numpy(transformed).unsqueeze(0).to(device))
        depth_pred = torch.nn.functional.interpolate(depth_pred.unsqueeze(1), size=(512, 512), mode="bicubic").squeeze()

    depth_map = torch.clamp(depth_pred, 0, 3000) / 3000.0
    depth_map = depth_map.unsqueeze(0).unsqueeze(0).to(device)

    merged_tensor = torch.cat([img_tensor, depth_map], dim=1)
    return merged_tensor, (orig_w, orig_h)

# ------------------- Infer ----------------------
def infer(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        gamma = torch.tensor([2.2], device=device)
        deflare_img, _, _ = predict_flare_from_6_channel(output, gamma)
    return deflare_img

# ------------------- Exposure Correction ----------------------
def auto_exposure_correction(image_tensor):
    img_np = image_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    corrected = img_np

    if mean_brightness < 80:  # Dark
        corrected = np.power(corrected, 1 / 1.5)
    elif mean_brightness > 180:  # Bright
        corrected = np.power(corrected, 1.2)

    corrected = np.clip(corrected, 0, 1)
    return torch.tensor(corrected).permute(2, 0, 1).unsqueeze(0).to(device)

# ------------------- Streamlit UI ----------------------

st.title("NoctiVision - A Flare Removal AI 🌠")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Original", use_container_width=True)

    with st.spinner("Processing..."):
        merged, original_size = preprocess(img_pil)
        output_img = infer(merged)
        output_img = auto_exposure_correction(output_img)

        # Resize output to original dimensions
        output_np = output_img.cpu().squeeze().permute(1, 2, 0).numpy()
        output_np_resized = cv2.resize(output_np, original_size)

    st.image(output_np_resized, caption="Flare-Free & Exposure Corrected", use_container_width=True)

    # Download
    output_pil = Image.fromarray((output_np_resized * 255).astype(np.uint8))
    st.download_button("Download", data=output_pil.tobytes(), file_name="deflared.png", mime="image/png")
