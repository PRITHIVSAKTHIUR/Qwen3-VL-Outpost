import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Qwen2.5-VL-7B-Instruct
MODEL_ID_M = "Qwen/Qwen2.5-VL-7B-Instruct"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Qwen2.5-VL-3B-Instruct
MODEL_ID_X = "Qwen/Qwen2.5-VL-3B-Instruct"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Qwen2.5-VL-7B-Abliterated-Caption-it
MODEL_ID_Q = "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it"
processor_q = AutoProcessor.from_pretrained(MODEL_ID_Q, trust_remote_code=True)
model_q = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Lumian2-VLR-7B-Thinking
MODEL_ID_Y = "prithivMLmods/Lumian2-VLR-7B-Thinking"
processor_y = AutoProcessor.from_pretrained(MODEL_ID_Y, trust_remote_code=True)
model_y = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Y,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "Qwen2.5-VL-7B-Instruct":
        processor = processor_m
        model = model_m
    elif model_name == "Qwen2.5-VL-3B-Instruct":
        processor = processor_x
        model = model_x
    elif model_name == "Qwen2.5-VL-7B-Abliterated-Caption-it":
        processor = processor_q
        model = model_q
    elif model_name == "Lumian2-VLR-7B-Thinking":
        processor = processor_y
        model = model_y
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "Qwen2.5-VL-7B-Instruct":
        processor = processor_m
        model = model_m
    elif model_name == "Qwen2.5-VL-3B-Instruct":
        processor = processor_x
        model = model_x
    elif model_name == "Qwen2.5-VL-7B-Abliterated-Caption-it":
        processor = processor_q
        model = model_q
    elif model_name == "Lumian2-VLR-7B-Thinking":
        processor = processor_y
        model = model_y
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

# Define examples for image and video inference
image_examples = [
    ["Explain the content in detail.", "images/D.jpg"],
    ["Explain the content (ocr).", "images/O.jpg"],
    ["What is the core meaning of the poem?", "images/S.jpg"],
    ["Provide a detailed caption for the image.", "images/A.jpg"],
    ["Explain the pie-chart in detail.", "images/2.jpg"],
    ["Jsonify Data.", "images/1.jpg"],
]

video_examples = [
    ["Explain the ad in detail", "videos/1.mp4"],
    ["Identify the main actions in the video", "videos/2.mp4"],
    ["Identify the main scenes in the video", "videos/3.mp4"]
]

css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
.canvas-output {
    border: 2px solid #4682B4;
    border-radius: 10px;
    padding: 20px;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[Qwen2.5-VL-Outpost](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="✦︎ Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    ) 
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
                
        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Output")
                output = gr.Textbox(label="Raw Output", interactive=False, lines=3, scale=2)
            
                with gr.Accordion("(Result.md)", open=False):
                    markdown_output = gr.Markdown()
                
            model_choice = gr.Radio(
                choices=["Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-3B-Instruct", "Lumian2-VLR-7B-Thinking", "Qwen2.5-VL-7B-Abliterated-Caption-it"],
                label="Select Model",
                value="Qwen2.5-VL-7B-Instruct"
            )
            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Qwen2.5-VL/discussions)")
            
            gr.Markdown(
                """
            > [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): The Qwen2.5-VL-7B-Instruct model is a multimodal AI model developed by Alibaba Cloud that excels at understanding both text and images. It's a Vision-Language Model (VLM) designed to handle various visual understanding tasks, including image understanding, video analysis, and even multilingual support.  
            >
            > [Qwen2.5-VL-7B-Abliterated-Caption-it](prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it): Qwen2.5-VL-7B-Abliterated-Caption-it is a fine-tuned version of Qwen2.5-VL-7B-Instruct, optimized for Abliterated Captioning / Uncensored Captioning. This model excels at generating detailed, context-rich, and high-fidelity captions across diverse image categories and variational aspect ratios, offering robust visual understanding without filtering or censorship.  
                """
            )

            gr.Markdown("> [Lumian2-VLR-7B-Thinking](https://huggingface.co/prithivMLmods/Lumian2-VLR-7B-Thinking): The Lumian2-VLR-7B-Thinking model is a high-fidelity vision-language reasoning (experimental model) system designed for fine-grained multimodal understanding. Built on Qwen2.5-VL-7B-Instruct, this model enhances image captioning, sampled video reasoning, and document comprehension through explicit grounded reasoning. It produces structured reasoning traces aligned with visual coordinates, enabling explainable multimodal reasoning.")
        
            gr.Markdown(">⚠️note: all the models in space are not guaranteed to perform well in video inference use cases.")
            
    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)
