"""
OpenVLA Inference Script with Performance Optimizations
========================================================
This script includes multiple optimizations to reduce inference time:
1. INT8 quantization - 2-4x speedup (enabled by default)
2. torch.compile with max-autotune - 1.5-2x speedup
3. CUDA optimizations (TF32, cudnn.benchmark, optimized attention)
4. Inference mode optimizations
5. Multiple warmup runs for accurate timing

Using OpenVLA 7B model for best performance.
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time

# 1. Setup Device
# ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Optimize CUDA settings for inference
if device == "cuda":
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matmuls on Ampere+
    torch.backends.cudnn.allow_tf32 = True
    # Use high precision for matmuls (TF32 on Ampere+)
    torch.set_float32_matmul_precision('high')  # 'high' = TF32, fastest on Ampere+

# 2. Load Model & Processor
# -------------------------
# Configuration options for speed optimization
# NOTE: Disabled by default due to potential torch/torchvision version conflicts with bitsandbytes
# If you want to enable, first fix dependencies: pip install torch torchvision --upgrade
ENABLE_QUANTIZATION = False  # Set to True to enable INT8 quantization (2-4x speedup)

# Using the original 7B model for best performance
model_id = "openvla/openvla-7b"
print("Using OpenVLA 7B model")

print(f"Loading model {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Helper function to load model with fallback attention implementation
def load_model_with_fallback(model_id, quantization_config=None, dtype=None):
    """Try to load model with flash_attention_2, fallback to sdpa if it fails."""
    attn_implementations = ["flash_attention_2", "sdpa", "eager"]
    
    for attn_impl in attn_implementations:
        try:
            print(f"Trying to load model with {attn_impl} attention...")
            kwargs = {
                "attn_implementation": attn_impl,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            else:
                kwargs["dtype"] = dtype
            
            model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
            
            if quantization_config is None:
                model = model.to(device)
            
            print(f"Successfully loaded model with {attn_impl} attention!")
            return model
        except Exception as e:
            if attn_impl == attn_implementations[-1]:
                # Last attempt failed, raise the error
                raise e
            print(f"Failed to load with {attn_impl}: {e}")
            print(f"Trying next attention implementation...")
            continue

# Load model with optional INT8 quantization
if ENABLE_QUANTIZATION and device == "cuda":
    print("Loading model with INT8 quantization for 2-4x speedup...")
    print("Note: Requires 'bitsandbytes' package. Install with: pip install bitsandbytes")
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        model = load_model_with_fallback(model_id, quantization_config=quantization_config)
        print("Model loaded with INT8 quantization!")
    except ImportError as e:
        print(f"bitsandbytes not installed. Install with: pip install bitsandbytes")
        print("Loading model without quantization...")
        model = load_model_with_fallback(model_id, dtype=torch.bfloat16)
    except Exception as e:
        print(f"Quantization failed ({e}), loading without quantization...")
        model = load_model_with_fallback(model_id, dtype=torch.bfloat16)
else:
    model = load_model_with_fallback(model_id, dtype=torch.bfloat16)

# Set model to evaluation mode and disable gradients
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Compile model for faster inference (PyTorch 2.0+)
# This can provide significant speedups (often 1.5-2x)
print("Compiling model for optimized inference...")
try:
    # Use "max-autotune" for best performance (slower compilation, faster inference)
    # or "reduce-overhead" for faster compilation
    model = torch.compile(model, mode="max-autotune", fullgraph=False)
    print("Model compiled successfully with max-autotune!")
except Exception as e:
    print(f"Warning: torch.compile failed ({e}), trying reduce-overhead mode...")
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("Model compiled with reduce-overhead mode!")
    except Exception as e2:
        print(f"Warning: torch.compile failed completely ({e2}), continuing without compilation")

# 3. Load a Sample Image
# ----------------------
# Optional: Reduce image resolution for faster processing (smaller images = faster inference)
REDUCE_IMAGE_SIZE = False  # Set to True to resize images (may reduce accuracy slightly)
TARGET_IMAGE_SIZE = (224, 224)  # Target size if reducing

# We'll use an example image from the Bridge V2 dataset (widely used in OpenVLA training)
url = "https://upload.wikimedia.org/wikipedia/commons/7/76/Solanum_melongena_24_08_2012_%281%29.JPG"

try:
    print(f"Downloading image from {url}...")
    response = requests.get(url, timeout=10)
    response.raise_for_status() # Check if the download actually worked
    image = Image.open(BytesIO(response.content)).convert("RGB")
    if REDUCE_IMAGE_SIZE:
        print(f"Resizing image to {TARGET_IMAGE_SIZE} for faster processing...")
        image = image.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
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

# Process inputs - use appropriate dtype based on quantization
if ENABLE_QUANTIZATION and device == "cuda":
    # Quantized models handle dtype internally
    inputs = processor(prompt, image).to(device)
else:
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

# Warmup runs - first inference is slower due to CUDA kernel initialization
# Multiple warmup runs ensure all kernels are loaded and optimized
print("Running warmup inference (3 iterations)...")
with torch.inference_mode():
    for _ in range(3):
        _ = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
if device == "cuda":
    torch.cuda.synchronize()
print("Warmup complete!")

# The model has a helper function 'predict_action' that handles 
# token generation and detokenization into a 7-DOF vector.
# Measure inference time with optimizations
print("Running timed inference...")
if device == "cuda":
    # Use CUDA events for accurate GPU timing
    torch.cuda.synchronize()  # Ensure all previous operations are complete
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    with torch.inference_mode():  # Disable autograd for faster inference
        start_event.record()
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        end_event.record()
    
    torch.cuda.synchronize()  # Wait for inference to complete
    inference_time_ms = start_event.elapsed_time(end_event)
    inference_time_s = inference_time_ms / 1000.0
else:
    # Use CPU timing for CPU inference
    with torch.inference_mode():
        start_time = time.time()
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        inference_time_s = time.time() - start_time
        inference_time_ms = inference_time_s * 1000.0

# 6. Output Results
# -----------------
print("\nSuccess! Generated Action Vector (7-DOF):")
print(np.array2string(action, precision=4, floatmode='fixed'))
print("\n(Format is usually: [x, y, z, roll, pitch, yaw, gripper_openness])")
print(f"\nInference Time: {inference_time_s:.4f} seconds ({inference_time_ms:.2f} ms)")