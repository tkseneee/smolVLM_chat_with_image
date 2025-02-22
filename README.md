SmolVLM-500M: Lightweight Vision-Language Model for Image Understanding

📌 Overview
This project showcases SmolVLM-500M-Instruct, a lightweight Vision-Language Model (VLM) that can process images and answer questions in a conversational manner.
✅ Small Model (~500M parameters) – Low memory usage
✅ Runs on CPU & GPU – No high-end hardware needed
✅ Multi-turn Q&A – Handles follow-up queries
✅ Edge Device Compatible – Ideal for real-time inference

📷 Example: Convocation Event Analysis
The model was tested on a university convocation image, where it successfully:
- Identified people and their gender
- Answered context-based questions (e.g., "Who is receiving a certificate?")
- Handled follow-up queries (e.g., "How many faces are not visible?")

🚀 Installation

1️⃣ Clone the Repository
```
git clone https://github.com/your-username/smolVLM-500M.git
cd smolVLM-500M
```

2️⃣ Install Dependencies
Ensure PyTorch and Hugging Face Transformers are installed:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
```
For CPU-only installation, remove --index-url.

3️⃣ Install FlashAttention (Optional for Faster GPU Inference)
```
pip install flash-attn --no-build-isolation
```

🏃 Usage

Load the Model & Processor
```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Initialize processor & model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Load image
image = Image.open("test_image.jpg")  # Replace with your image file

# Create conversation
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "How many people are in this image?"}
    ]}
]

# Process & generate response
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print("Assistant:", generated_text[0])
```

📌 Model Capabilities
✔️ Counts people in an image
✔️ Identifies objects and interactions
✔️ Answers follow-up visual questions

🎯 Future Work
🔹 Fine-tuning for specific tasks
🔹 Deployment on edge devices (Jetson, Raspberry Pi)
🔹 Integration into a chatbot for interactive Q&A


