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
    model.load_state_dict(torch.load(model_path))
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

# Load models early
model = load_model(r"C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\final train\final_trained_model.pth", output_ch=6)
depth_model = load_depth_model(r"C:\Users\sweet\Downloads\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\Flare-Free-Vision-Empowering-Uformer-with-Depth-Insights-main\DPT\dpt_hybrid-midas-501f0c75.pt")

# ------------------- Utility Functions ----------------------

def preprocess(img_pil, depth_model):
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
    original_img = img_tensor.clone()
    img_np = np.uint8(img_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255)

    img_input = np.zeros((1, 3, 384, 384), dtype=np.float32)
    img_input[0] = depth_transform({"image": img_np[0]})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device)
        prediction = depth_model(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=512,
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    prediction = np.clip(prediction, 0, 3000)
    depth_map = torch.from_numpy(prediction / 3000).unsqueeze(0).unsqueeze(0).to(device)
    merge_img = torch.cat((original_img, depth_map), dim=1)
    return merge_img, original_img

def infer(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        gamma = torch.tensor([2.2], device=device)
        deflare_img, flare_img_predicted, _ = predict_flare_from_6_channel(output, gamma)
    return deflare_img

def auto_exposure_correction(image_tensor):
    img = image_tensor.clone().cpu()
    img_np = img.squeeze().permute(1, 2, 0).numpy()

    # Convert to grayscale for brightness estimation
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)

    # Thresholds
    too_dark = mean_brightness < 80
    too_bright = mean_brightness > 180

    corrected = img_np

    if too_dark:
        gamma = 1.5  # brighten
        corrected = np.power(corrected, 1 / gamma)
    elif too_bright:
        gamma = 1.2  # darken
        corrected = np.power(corrected, gamma)

    corrected = np.clip(corrected, 0, 1)
    corrected_tensor = torch.tensor(corrected).permute(2, 0, 1).unsqueeze(0).to(device).float()
    return corrected_tensor

# ------------------- Streamlit UI ----------------------

st.title("Flare Removal UI 🌠")
st.write("Upload a flared image and get a flare-free version with automatic exposure correction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Removing flare and correcting exposure..."):
        try:
            merge_img, original_img = preprocess(image_pil, depth_model)
            output_img = infer(merge_img)
            output_img = auto_exposure_correction(output_img)

            output_np = output_img.cpu().squeeze().permute(1, 2, 0).numpy()
            st.image(output_np, caption="Flare-Free & Exposure-Corrected Output", use_column_width=True)

            output_pil = transforms.ToPILImage()(output_img.squeeze().cpu())
            st.download_button("Download Result", data=output_pil.tobytes(), file_name="deflared_output.png")
        except Exception as e:
            st.error(f"Error: {str(e)}")
