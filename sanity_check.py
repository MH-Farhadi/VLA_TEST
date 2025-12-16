import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# 1. Setup Device
# ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# 2. Load Model & Processor
# -------------------------
# Note: trust_remote_code=True is required because OpenVLA uses custom modeling code.
# We use bfloat16 for efficiency on modern GPUs (Ampere/Ada).
model_id = "openvla/openvla-7b"

print(f"Loading model {model_id}... (this might take a moment to download ~15GB)")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",  # Uses the library you just installed
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# 3. Load a Sample Image
# ----------------------
# We'll use an example image from the Bridge V2 dataset (widely used in OpenVLA training)
url = "https://upload.wikimedia.org/wikipedia/commons/7/76/Solanum_melongena_24_08_2012_%281%29.JPG"

try:
    print(f"Downloading image from {url}...")
    response = requests.get(url, timeout=10)
    response.raise_for_status() # Check if the download actually worked
    image = Image.open(BytesIO(response.content)).convert("RGB")
except Exception as e:
    print(f"Image download failed: {e}")
    print("Creating a dummy image instead (output will be nonsense, but code will run).")
    image = Image.new('RGB', (224, 224), color='green')

# 4. Define the Prompt
# --------------------
# OpenVLA expects a specific prompt format:
# "In: What action should the robot take to {task}?\nOut:"
prompt = "In: What action should the robot take to put the eggplant in the pot?\nOut:"

# 5. Run Inference
# ----------------
print(f"Processing prompt: '{prompt}'")
inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

# The model has a helper function 'predict_action' that handles 
# token generation and detokenization into a 7-DOF vector.
action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 6. Output Results
# -----------------
print("\nSuccess! Generated Action Vector (7-DOF):")
print(np.array2string(action, precision=4, floatmode='fixed'))
print("\n(Format is usually: [x, y, z, roll, pitch, yaw, gripper_openness])")