import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, Iterable

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2
import requests
import fitz

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

# --- Theme and CSS Definition ---

# Define the new OrangeRed color palette
colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",  # OrangeRed base color
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red, # Use the new color
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

# Instantiate the new theme
orange_red_theme = OrangeRedTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# --- Model Loading ---

# Load Qwen3-VL-4B-Instruct
MODEL_ID_Q4B = "Qwen/Qwen3-VL-4B-Instruct"
processor_q4b = AutoProcessor.from_pretrained(MODEL_ID_Q4B, trust_remote_code=True)
model_q4b = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q4B,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

# Load Qwen3-VL-8B-Instruct
MODEL_ID_Q8B = "Qwen/Qwen3-VL-8B-Instruct"
processor_q8b = AutoProcessor.from_pretrained(MODEL_ID_Q8B, trust_remote_code=True)
model_q8b = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q8B,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

# Load Qwen3-VL-2B-Instruct
MODEL_ID_Q2B = "Qwen/Qwen3-VL-2B-Instruct"
processor_q2b = AutoProcessor.from_pretrained(MODEL_ID_Q2B, trust_remote_code=True)
model_q2b = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q2B,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

# Load Qwen2.5-VL-7B-Instruct
MODEL_ID_M7B = "Qwen/Qwen2.5-VL-7B-Instruct"
processor_m7b = AutoProcessor.from_pretrained(MODEL_ID_M7B, trust_remote_code=True)
model_m7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M7B,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Qwen2.5-VL-3B-Instruct
MODEL_ID_X3B = "Qwen/Qwen2.5-VL-3B-Instruct"
processor_x3b = AutoProcessor.from_pretrained(MODEL_ID_X3B, trust_remote_code=True)
model_x3b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X3B,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()


# --- Helper Functions ---

def select_model(model_name: str):
    if model_name == "Qwen3-VL-4B-Instruct":
        return processor_q4b, model_q4b
    elif model_name == "Qwen3-VL-8B-Instruct":
        return processor_q8b, model_q8b
    elif model_name == "Qwen3-VL-2B-Instruct":
        return processor_q2b, model_q2b
    elif model_name == "Qwen2.5-VL-7B-Instruct":
        return processor_m7b, model_m7b
    elif model_name == "Qwen2.5-VL-3B-Instruct":
        return processor_x3b, model_x3b
    else:
        raise ValueError("Invalid model selected.")

def extract_gif_frames(gif_path: str):
    if not gif_path:
        return []
    with Image.open(gif_path) as gif:
        total_frames = gif.n_frames
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
        frames = []
        for i in frame_indices:
            gif.seek(i)
            frames.append(gif.convert("RGB").copy())
    return frames

def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frames.append(pil_image)
    vidcap.release()
    return frames

def convert_pdf_to_images(file_path: str, dpi: int = 200):
    if not file_path:
        return []
    images = []
    pdf_document = fitz.open(file_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(Image.open(BytesIO(img_data)))
    pdf_document.close()
    return images

def get_initial_pdf_state() -> Dict[str, Any]:
    return {"pages": [], "total_pages": 0, "current_page_index": 0}

def load_and_preview_pdf(file_path: Optional[str]) -> Tuple[Optional[Image.Image], Dict[str, Any], str]:
    state = get_initial_pdf_state()
    if not file_path:
        return None, state, '<div style="text-align:center;">No file loaded</div>'
    try:
        pages = convert_pdf_to_images(file_path)
        if not pages:
            return None, state, '<div style="text-align:center;">Could not load file</div>'
        state["pages"] = pages
        state["total_pages"] = len(pages)
        page_info_html = f'<div style="text-align:center;">Page 1 / {state["total_pages"]}</div>'
        return pages[0], state, page_info_html
    except Exception as e:
        return None, state, f'<div style="text-align:center;">Failed to load preview: {e}</div>'

def navigate_pdf_page(direction: str, state: Dict[str, Any]):
    if not state or not state["pages"]:
        return None, state, '<div style="text-align:center;">No file loaded</div>'
    current_index = state["current_page_index"]
    total_pages = state["total_pages"]
    if direction == "prev":
        new_index = max(0, current_index - 1)
    elif direction == "next":
        new_index = min(total_pages - 1, current_index + 1)
    else:
        new_index = current_index
    state["current_page_index"] = new_index
    image_preview = state["pages"][new_index]
    page_info_html = f'<div style="text-align:center;">Page {new_index + 1} / {total_pages}</div>'
    return image_preview, state, page_info_html

# --- Generation Functions ---

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return
    try:
        processor, model = select_model(model_name)
    except ValueError as e:
        yield str(e), str(e)
        return
        
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
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
def generate_video(model_name: str, text: str, video_path: str, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return
    try:
        processor, model = select_model(model_name)
    except ValueError as e:
        yield str(e), str(e)
        return

    frames = downsample_video(video_path)
    if not frames:
        yield "Could not process video.", "Could not process video."
        return
        
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    for frame in frames:
        messages[0]["content"].insert(0, {"type": "image"})
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_full], images=frames, return_tensors="pt", padding=True).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
       # buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_pdf(model_name: str, text: str, state: Dict[str, Any], max_new_tokens: int = 2048, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if not state or not state["pages"]:
        yield "Please upload a PDF file first.", "Please upload a PDF file first."
        return
    try:
        processor, model = select_model(model_name)
    except ValueError as e:
        yield str(e), str(e)
        return

    page_images = state["pages"]
    full_response = ""
    for i, image in enumerate(page_images):
        page_header = f"--- Page {i+1}/{len(page_images)} ---\n"
        yield full_response + page_header, full_response + page_header
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        page_buffer = ""
        for new_text in streamer:
            page_buffer += new_text
            yield full_response + page_header + page_buffer, full_response + page_header + page_buffer
            time.sleep(0.01)
        full_response += page_header + page_buffer + "\n\n"

@spaces.GPU
def generate_caption(model_name: str, image: Image.Image, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if image is None:
        yield "Please upload an image to caption.", "Please upload an image to caption."
        return
    try:
        processor, model = select_model(model_name)
    except ValueError as e:
        yield str(e), str(e)
        return
        
    system_prompt = (
        "You are an AI assistant. For the given image, write a precise caption and provide a structured set of "
        "attributes describing visual elements like objects, people, actions, colors, and environment."
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": system_prompt}]}]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)
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
def generate_gif(model_name: str, text: str, gif_path: str, max_new_tokens: int = 1024, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.2):
    if gif_path is None:
        yield "Please upload a GIF.", "Please upload a GIF."
        return
    try:
        processor, model = select_model(model_name)
    except ValueError as e:
        yield str(e), str(e)
        return

    frames = extract_gif_frames(gif_path)
    if not frames:
        yield "Could not process GIF.", "Could not process GIF."
        return
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    for frame in frames:
        messages[0]["content"].insert(0, {"type": "image"})
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_full], images=frames, return_tensors="pt", padding=True).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens, "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k, "repetition_penalty": repetition_penalty}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
      # buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

# --- Examples and Gradio UI ---

image_examples = [["Perform OCR on the image...", "examples/images/1.jpg"],
                  ["Caption the image. Describe the safety measures shown in the image. Conclude whether the situation is (safe or unsafe)...", "examples/images/2.jpg"],
                  ["Solve the problem...", "examples/images/3.png"]]
video_examples = [["Explain the Ad video in detail.", "examples/videos/1.mp4"],
                  ["Explain the video in detail.", "examples/videos/2.mp4"]]
pdf_examples = [["Extract the content precisely.", "examples/pdfs/doc1.pdf"],
                ["Analyze and provide a short report.", "examples/pdfs/doc2.pdf"]]
gif_examples = [["Describe this GIF.", "examples/gifs/1.gif"],
                ["Describe this GIF.", "examples/gifs/2.gif"]]
caption_examples = [["examples/captions/1.JPG"],
                    ["examples/captions/2.jpeg"], ["examples/captions/3.jpeg"]]

with gr.Blocks(theme=orange_red_theme, css=css) as demo:
    pdf_state = gr.State(value=get_initial_pdf_state())
    gr.Markdown("# **Qwen3-VL-Outpost**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Upload Image", height=290)
                    image_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=image_examples, inputs=[image_query, image_upload])

                with gr.TabItem("PDF Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_query = gr.Textbox(label="Query Input", placeholder="e.g., 'Summarize this document'")
                            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                            pdf_submit = gr.Button("Submit", variant="primary")
                        with gr.Column(scale=1):
                            pdf_preview_img = gr.Image(label="PDF Preview", height=290)
                            with gr.Row():
                                prev_page_btn = gr.Button("◀ Previous")
                                page_info = gr.HTML('<div style="text-align:center;">No file loaded</div>')
                                next_page_btn = gr.Button("Next ▶")
                    gr.Examples(examples=pdf_examples, inputs=[pdf_query, pdf_upload])

                with gr.TabItem("Long Caption"):
                    caption_image_upload = gr.Image(type="pil", label="Image to Caption", height=290)
                    caption_submit = gr.Button("Generate Caption", variant="primary")
                    gr.Examples(examples=caption_examples, inputs=[caption_image_upload])
                    
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Upload Video(≤30s)", height=290)
                    video_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=video_examples, inputs=[video_query, video_upload])
                    
                with gr.TabItem("Gif Inference"):
                    gif_query = gr.Textbox(label="Query Input", placeholder="e.g., 'What is happening in this gif?'")
                    gif_upload = gr.Image(type="filepath", label="Upload GIF", height=290)
                    gif_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(examples=gif_examples, inputs=[gif_query, gif_upload])
                    
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=12, show_copy_button=True)
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown(label="(Result.Md)", latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False}
                            ])

            model_choice = gr.Radio(
                choices=[
                    "Qwen3-VL-4B-Instruct",
                    "Qwen3-VL-8B-Instruct",
                    "Qwen3-VL-2B-Instruct",
                    "Qwen2.5-VL-7B-Instruct",
                    "Qwen2.5-VL-3B-Instruct"
                ],
                label="Select Model",
                value="Qwen3-VL-4B-Instruct"
            )

    # --- Event Handlers ---
    
    image_submit.click(fn=generate_image,
                       inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                       outputs=[output, markdown_output])
    
    video_submit.click(fn=generate_video,
                       inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                       outputs=[output, markdown_output])
    
    pdf_submit.click(fn=generate_pdf,
                     inputs=[model_choice, pdf_query, pdf_state, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                     outputs=[output, markdown_output])
                     
    gif_submit.click(fn=generate_gif,
                     inputs=[model_choice, gif_query, gif_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                     outputs=[output, markdown_output])
                     
    caption_submit.click(fn=generate_caption,
                         inputs=[model_choice, caption_image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
                         outputs=[output, markdown_output])

    pdf_upload.change(fn=load_and_preview_pdf, inputs=[pdf_upload], outputs=[pdf_preview_img, pdf_state, page_info])
    prev_page_btn.click(fn=lambda s: navigate_pdf_page("prev", s), inputs=[pdf_state], outputs=[pdf_preview_img, pdf_state, page_info])
    next_page_btn.click(fn=lambda s: navigate_pdf_page("next", s), inputs=[pdf_state], outputs=[pdf_preview_img, pdf_state, page_info])

if __name__ == "__main__":
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)
