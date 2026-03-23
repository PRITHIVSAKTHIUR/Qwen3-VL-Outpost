import os
import gc
import json
import time
import base64
from io import BytesIO
from threading import Thread

import gradio as gr
import spaces
import torch
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_ID_Q4B = "Qwen/Qwen3-VL-4B-Instruct"
processor_q4b = AutoProcessor.from_pretrained(MODEL_ID_Q4B, trust_remote_code=True)
model_q4b = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q4B,
    attn_implementation="kernels-community/flash-attn3",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

MODEL_ID_Q8B = "Qwen/Qwen3-VL-8B-Instruct"
processor_q8b = AutoProcessor.from_pretrained(MODEL_ID_Q8B, trust_remote_code=True)
model_q8b = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q8B,
    attn_implementation="kernels-community/flash-attn3",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

MODEL_ID_Q2B = "Qwen/Qwen3-VL-2B-Instruct"
processor_q2b = AutoProcessor.from_pretrained(MODEL_ID_Q2B, trust_remote_code=True)
model_q2b = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_Q2B,
    attn_implementation="kernels-community/flash-attn3",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

MODEL_ID_M7B = "Qwen/Qwen2.5-VL-7B-Instruct"
processor_m7b = AutoProcessor.from_pretrained(MODEL_ID_M7B, trust_remote_code=True)
model_m7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M7B,
    attn_implementation="kernels-community/flash-attn3",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_X3B = "Qwen/Qwen2.5-VL-3B-Instruct"
processor_x3b = AutoProcessor.from_pretrained(MODEL_ID_X3B, trust_remote_code=True)
model_x3b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X3B,
    attn_implementation="kernels-community/flash-attn3",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_MAP = {
    "Qwen3-VL-4B-Instruct": (processor_q4b, model_q4b),
    "Qwen3-VL-8B-Instruct": (processor_q8b, model_q8b),
    "Qwen3-VL-2B-Instruct": (processor_q2b, model_q2b),
    "Qwen2.5-VL-7B-Instruct": (processor_m7b, model_m7b),
    "Qwen2.5-VL-3B-Instruct": (processor_x3b, model_x3b),
}
MODEL_CHOICES = list(MODEL_MAP.keys())

image_examples = [
    {"query": "Perform OCR on the image...", "media": "examples/images/1.jpg", "model": "Qwen3-VL-4B-Instruct"},
    {"query": "Caption the image. Describe the safety measures shown in the image. Conclude whether the situation is safe or unsafe.", "media": "examples/images/2.jpg", "model": "Qwen3-VL-8B-Instruct"},
    {"query": "Solve the problem...", "media": "examples/images/3.png", "model": "Qwen3-VL-2B-Instruct"},
]


def select_model(model_name: str):
    if model_name not in MODEL_MAP:
        raise ValueError("Invalid model selected.")
    return MODEL_MAP[model_name]


def pil_to_data_url(img: Image.Image, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def file_to_data_url(path):
    if not os.path.exists(path):
        return ""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


def make_thumb_b64(path, max_dim=240):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim))
        return pil_to_data_url(img, "JPEG")
    except Exception as e:
        print("Thumbnail error:", e)
        return ""


def build_example_cards_html():
    cards = ""
    for i, ex in enumerate(image_examples):
        thumb = make_thumb_b64(ex["media"])
        prompt_short = ex["query"][:72] + ("..." if len(ex["query"]) > 72 else "")
        cards += f"""
        <div class="example-card" data-idx="{i}">
            <div class="example-thumb-wrap">
                {"<img src='" + thumb + "' alt=''>" if thumb else "<div class='example-thumb-placeholder'>Preview</div>"}
                <div class="example-media-chip">IMAGE</div>
            </div>
            <div class="example-meta-row">
                <span class="example-badge">{ex["model"]}</span>
            </div>
            <div class="example-prompt-text">{prompt_short}</div>
        </div>
        """
    return cards


EXAMPLE_CARDS_HTML = build_example_cards_html()


def load_example_data(idx_str):
    try:
        idx = int(str(idx_str).strip())
    except Exception:
        return gr.update(value="")

    if idx < 0 or idx >= len(image_examples):
        return gr.update(value="")

    ex = image_examples[idx]
    media_b64 = file_to_data_url(ex["media"])
    if not media_b64:
        return gr.update(value=json.dumps({"status": "error", "message": "Could not load example image"}))

    return gr.update(value=json.dumps({
        "status": "ok",
        "query": ex["query"],
        "media": media_b64,
        "model": ex["model"],
        "name": os.path.basename(ex["media"]),
    }))


def b64_to_pil(b64_str):
    if not b64_str:
        return None
    try:
        if b64_str.startswith("data:"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        return None


def calc_timeout_generic(*args, **kwargs):
    gpu_timeout = kwargs.get("gpu_timeout", None)
    if gpu_timeout is None and args:
        gpu_timeout = args[-1]
    try:
        return int(gpu_timeout)
    except Exception:
        return 60


@spaces.GPU(duration=calc_timeout_generic)
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024, temperature: float = 0.6,
                   top_p: float = 0.9, top_k: int = 50,
                   repetition_penalty: float = 1.2, gpu_timeout: int = 60):
    try:
        if image is None:
            yield "[ERROR] Please upload an image."
            return
        if not text or not str(text).strip():
            yield "[ERROR] Please enter your instruction."
            return

        processor, model = select_model(model_name)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[prompt_full],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)

        streamer = TextIteratorStreamer(
            processor.tokenizer if hasattr(processor, "tokenizer") else processor,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_error = {"error": None}

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
        }

        def _run_generation():
            try:
                model.generate(**generation_kwargs)
            except Exception as e:
                generation_error["error"] = e
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = Thread(target=_run_generation, daemon=True)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            time.sleep(0.01)
            yield buffer

        thread.join(timeout=1.0)

        if generation_error["error"] is not None:
            err_msg = f"[ERROR] Inference failed: {str(generation_error['error'])}"
            if buffer.strip():
                yield buffer + "\n\n" + err_msg
            else:
                yield err_msg
            return

        if not buffer.strip():
            yield "[ERROR] No output was generated."

    except Exception as e:
        yield f"[ERROR] {str(e)}"
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_router(model_name, text, image_b64, max_new_tokens_v, temperature_v, top_p_v, top_k_v, repetition_penalty_v, gpu_timeout_v):
    try:
        image = b64_to_pil(image_b64)
        yield from generate_image(
            model_name=model_name,
            text=text,
            image=image,
            max_new_tokens=max_new_tokens_v,
            temperature=temperature_v,
            top_p=top_p_v,
            top_k=top_k_v,
            repetition_penalty=repetition_penalty_v,
            gpu_timeout=gpu_timeout_v,
        )
    except Exception as e:
        yield f"[ERROR] {str(e)}"


def noop():
    return None


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow-x:hidden}
body,.gradio-container{
    background:#0f0f13!important;
    font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    font-size:14px!important;color:#e4e4e7!important;min-height:100vh;overflow-x:hidden;
}
.dark body,.dark .gradio-container{background:#0f0f13!important;color:#e4e4e7!important}
footer{display:none!important}
.hidden-input{display:none!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important}

#gradio-run-btn,#example-load-btn{
    position:absolute!important;left:-9999px!important;top:-9999px!important;
    width:1px!important;height:1px!important;opacity:0.01!important;
    pointer-events:none!important;overflow:hidden!important;
}

.app-shell{
    background:#18181b;border:1px solid #27272a;border-radius:16px;
    margin:12px auto;max-width:1450px;overflow:hidden;
    box-shadow:0 25px 50px -12px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.03);
}
.app-header{
    background:linear-gradient(135deg,#18181b,#1e1e24);border-bottom:1px solid #27272a;
    padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
}
.app-header-left{display:flex;align-items:center;gap:12px}
.app-logo{
    width:38px;height:38px;background:linear-gradient(135deg,#0000CD,#2645ff,#5876ff);
    border-radius:10px;display:flex;align-items:center;justify-content:center;
    box-shadow:0 4px 12px rgba(0,0,205,.35);
}
.app-logo svg{width:22px;height:22px;fill:#fff;flex-shrink:0}
.app-title{
    font-size:18px;font-weight:700;background:linear-gradient(135deg,#f5f5f5,#bdbdbd);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.3px;
}
.app-badge{
    font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;
    background:rgba(0,0,205,.12);color:#8da1ff;border:1px solid rgba(0,0,205,.25);letter-spacing:.3px;
}
.app-badge.fast{background:rgba(38,69,255,.10);color:#90a3ff;border:1px solid rgba(38,69,255,.22)}

.model-tabs-bar{
    background:#18181b;border-bottom:1px solid #27272a;padding:10px 16px;
    display:flex;gap:8px;align-items:center;flex-wrap:wrap;
}
.model-tab{
    display:inline-flex;align-items:center;justify-content:center;gap:6px;
    min-width:32px;height:34px;background:transparent;border:1px solid #27272a;
    border-radius:999px;cursor:pointer;font-size:12px;font-weight:600;padding:0 12px;
    color:#ffffff!important;transition:all .15s ease;
}
.model-tab:hover{background:rgba(0,0,205,.12);border-color:rgba(0,0,205,.35)}
.model-tab.active{background:rgba(0,0,205,.22);border-color:#0000CD;color:#fff!important;box-shadow:0 0 0 2px rgba(0,0,205,.10)}
.model-tab-label{font-size:12px;color:#ffffff!important;font-weight:600}

.app-main-row{display:flex;gap:0;flex:1;overflow:hidden}
.app-main-left{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #27272a}
.app-main-right{width:500px;display:flex;flex-direction:column;flex-shrink:0;background:#18181b}

#media-drop-zone{
    position:relative;background:#09090b;height:440px;min-height:440px;max-height:440px;overflow:hidden;
}
#media-drop-zone.drag-over{outline:2px solid #0000CD;outline-offset:-2px;background:rgba(0,0,205,.04)}
.upload-prompt-modern{
    position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:20px;z-index:20;overflow:hidden;
}
.upload-click-area{
    display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;
    padding:28px 36px;max-width:92%;max-height:92%;border:2px dashed #3f3f46;border-radius:16px;
    background:rgba(0,0,205,.03);transition:all .2s ease;gap:8px;text-align:center;overflow:hidden;
}
.upload-click-area:hover{background:rgba(0,0,205,.08);border-color:#0000CD;transform:scale(1.02)}
.upload-click-area:active{background:rgba(0,0,205,.12);transform:scale(.99)}
.upload-click-area svg{width:86px;height:86px;max-width:100%;flex-shrink:0}
.upload-main-text{color:#a1a1aa;font-size:14px;font-weight:600;margin-top:4px}
.upload-sub-text{color:#71717a;font-size:12px}

.single-preview-wrap{
    width:100%;height:100%;display:none;align-items:center;justify-content:center;padding:16px;overflow:hidden;
}
.single-preview-card{
    width:100%;height:100%;max-width:100%;max-height:100%;border-radius:14px;overflow:hidden;border:1px solid #27272a;background:#111114;
    display:flex;align-items:center;justify-content:center;position:relative;
}
.single-preview-card img{
    width:100%;height:100%;max-width:100%;max-height:100%;object-fit:contain;display:block;background:#000;border:none;
}
.preview-overlay-actions{
    position:absolute;top:12px;right:12px;display:flex;gap:8px;z-index:5;
}
.preview-action-btn{
    display:inline-flex;align-items:center;justify-content:center;min-width:34px;height:34px;padding:0 12px;background:rgba(0,0,0,.65);
    border:1px solid rgba(255,255,255,.14);border-radius:10px;cursor:pointer;color:#fff!important;font-size:12px;font-weight:600;transition:all .15s ease;
}
.preview-action-btn:hover{background:#0000CD;border-color:#0000CD}

.hint-bar{
    background:rgba(0,0,205,.06);border-top:1px solid #27272a;border-bottom:1px solid #27272a;
    padding:10px 20px;font-size:13px;color:#a1a1aa;line-height:1.7;
}
.hint-bar b{color:#8da1ff;font-weight:600}
.hint-bar kbd{
    display:inline-block;padding:1px 6px;background:#27272a;border:1px solid #3f3f46;border-radius:4px;
    font-family:'JetBrains Mono',monospace;font-size:11px;color:#a1a1aa;
}

.examples-section{border-top:1px solid #27272a;padding:12px 16px}
.examples-title{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;
}
.examples-scroll{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}
.examples-scroll::-webkit-scrollbar{height:6px}
.examples-scroll::-webkit-scrollbar-track{background:#09090b;border-radius:3px}
.examples-scroll::-webkit-scrollbar-thumb{background:#27272a;border-radius:3px}
.examples-scroll::-webkit-scrollbar-thumb:hover{background:#3f3f46}
.example-card{
    position:relative;flex-shrink:0;width:220px;background:#09090b;border:1px solid #27272a;border-radius:10px;overflow:hidden;cursor:pointer;transition:all .2s ease;
}
.example-card:hover{border-color:#0000CD;transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,205,.15)}
.example-card.loading{opacity:.5;pointer-events:none}
.example-thumb-wrap{height:120px;overflow:hidden;background:#18181b;position:relative}
.example-thumb-wrap img{width:100%;height:100%;object-fit:cover}
.example-media-chip{
    position:absolute;top:8px;left:8px;display:inline-flex;padding:3px 7px;background:rgba(0,0,0,.7);border:1px solid rgba(255,255,255,.12);
    border-radius:999px;font-size:10px;font-weight:700;color:#fff;letter-spacing:.5px;
}
.example-thumb-placeholder{
    width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#18181b;color:#3f3f46;font-size:11px;
}
.example-meta-row{padding:6px 10px;display:flex;align-items:center;gap:6px}
.example-badge{
    display:inline-flex;padding:2px 7px;background:rgba(0,0,205,.12);border-radius:4px;font-size:10px;font-weight:600;color:#8da1ff;
    font-family:'JetBrains Mono',monospace;white-space:nowrap;
}
.example-prompt-text{
    padding:0 10px 8px;font-size:11px;color:#a1a1aa;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
}

.panel-card{border-bottom:1px solid #27272a}
.panel-card-title{
    padding:12px 20px;font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
}
.panel-card-body{padding:16px 20px;display:flex;flex-direction:column;gap:8px}
.modern-label{font-size:13px;font-weight:500;color:#a1a1aa;margin-bottom:4px;display:block}
.modern-textarea{
    width:100%;background:#09090b;border:1px solid #27272a;border-radius:8px;padding:10px 14px;font-family:'Inter',sans-serif;font-size:14px;color:#e4e4e7;
    resize:none;outline:none;min-height:100px;transition:border-color .2s;
}
.modern-textarea:focus{border-color:#0000CD;box-shadow:0 0 0 3px rgba(0,0,205,.15)}
.modern-textarea::placeholder{color:#3f3f46}
.modern-textarea.error-flash{
    border-color:#ef4444!important;box-shadow:0 0 0 3px rgba(239,68,68,.2)!important;animation:shake .4s ease;
}
@keyframes shake{0%,100%{transform:translateX(0)}20%,60%{transform:translateX(-4px)}40%,80%{transform:translateX(4px)}}

.toast-notification{
    position:fixed;top:24px;left:50%;transform:translateX(-50%) translateY(-120%);z-index:9999;padding:10px 24px;border-radius:10px;
    font-family:'Inter',sans-serif;font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px;box-shadow:0 8px 24px rgba(0,0,0,.5);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .35s ease;opacity:0;pointer-events:none;
}
.toast-notification.visible{transform:translateX(-50%) translateY(0);opacity:1;pointer-events:auto}
.toast-notification.error{background:linear-gradient(135deg,#dc2626,#b91c1c);color:#fff;border:1px solid rgba(255,255,255,.15)}
.toast-notification.warning{background:linear-gradient(135deg,#d97706,#b45309);color:#fff;border:1px solid rgba(255,255,255,.15)}
.toast-notification.info{background:linear-gradient(135deg,#1e40af,#1d4ed8);color:#fff;border:1px solid rgba(255,255,255,.15)}
.toast-notification .toast-icon{font-size:16px;line-height:1}
.toast-notification .toast-text{line-height:1.3}

.btn-run{
    display:flex;align-items:center;justify-content:center;gap:8px;width:100%;background:linear-gradient(135deg,#0000CD,#1638b7);border:none;border-radius:10px;
    padding:12px 24px;cursor:pointer;font-size:15px;font-weight:600;font-family:'Inter',sans-serif;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
    transition:all .2s ease;letter-spacing:-.2px;box-shadow:0 4px 16px rgba(0,0,205,.3),inset 0 1px 0 rgba(255,255,255,.1);
}
.btn-run:hover{
    background:linear-gradient(135deg,#2645ff,#0000CD);transform:translateY(-1px);box-shadow:0 6px 24px rgba(0,0,205,.45),inset 0 1px 0 rgba(255,255,255,.15);
}
.btn-run:active{transform:translateY(0);box-shadow:0 2px 8px rgba(0,0,205,.3)}
#custom-run-btn,#custom-run-btn *,#run-btn-label,.btn-run,.btn-run *{
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;fill:#ffffff!important;
}

.output-frame{border-bottom:1px solid #27272a;display:flex;flex-direction:column;position:relative}
.output-frame .out-title,.output-frame .out-title *,#output-title-label{
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
}
.output-frame .out-title{
    padding:10px 20px;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
    display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;
}
.out-title-right{display:flex;gap:8px;align-items:center}
.out-action-btn{
    display:inline-flex;align-items:center;justify-content:center;background:rgba(0,0,205,.1);border:1px solid rgba(0,0,205,.2);border-radius:6px;cursor:pointer;padding:3px 10px;
    font-size:11px;font-weight:500;color:#8da1ff!important;gap:4px;height:24px;transition:all .15s;
}
.out-action-btn:hover{background:rgba(0,0,205,.2);border-color:rgba(0,0,205,.35);color:#ffffff!important}
.out-action-btn svg{width:12px;height:12px;fill:#8da1ff}
.output-frame .out-body{
    flex:1;background:#09090b;display:flex;align-items:stretch;justify-content:stretch;overflow:hidden;min-height:320px;position:relative;
}
.output-scroll-wrap{width:100%;height:100%;padding:0;overflow:hidden}
.output-textarea{
    width:100%;height:320px;min-height:320px;max-height:320px;background:#09090b;color:#e4e4e7;border:none;outline:none;padding:16px 18px;font-size:13px;line-height:1.6;
    font-family:'JetBrains Mono',monospace;overflow:auto;resize:none;white-space:pre-wrap;
}
.output-textarea::placeholder{color:#52525b}
.output-textarea.error-flash{box-shadow:inset 0 0 0 2px rgba(239,68,68,.6)}
.modern-loader{
    display:none;position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(9,9,11,.92);z-index:15;flex-direction:column;align-items:center;justify-content:center;gap:16px;backdrop-filter:blur(4px);
}
.modern-loader.active{display:flex}
.modern-loader .loader-spinner{
    width:36px;height:36px;border:3px solid #27272a;border-top-color:#0000CD;border-radius:50%;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.modern-loader .loader-text{font-size:13px;color:#a1a1aa;font-weight:500}
.loader-bar-track{width:200px;height:4px;background:#27272a;border-radius:2px;overflow:hidden}
.loader-bar-fill{
    height:100%;background:linear-gradient(90deg,#0000CD,#4d6dff,#0000CD);background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite;border-radius:2px;
}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

.settings-group{border:1px solid #27272a;border-radius:10px;margin:12px 16px;padding:0;overflow:hidden}
.settings-group-title{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.8px;padding:10px 16px;border-bottom:1px solid #27272a;background:rgba(24,24,27,.5);
}
.settings-group-body{padding:14px 16px;display:flex;flex-direction:column;gap:12px}
.slider-row{display:flex;align-items:center;gap:10px;min-height:28px}
.slider-row label{font-size:13px;font-weight:500;color:#a1a1aa;min-width:118px;flex-shrink:0}
.slider-row input[type="range"]{
    flex:1;-webkit-appearance:none;appearance:none;height:6px;background:#27272a;border-radius:3px;outline:none;min-width:0;
}
.slider-row input[type="range"]::-webkit-slider-thumb{
    -webkit-appearance:none;width:16px;height:16px;background:linear-gradient(135deg,#0000CD,#1638b7);border-radius:50%;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,205,.4);transition:transform .15s;
}
.slider-row input[type="range"]::-webkit-slider-thumb:hover{transform:scale(1.2)}
.slider-row input[type="range"]::-moz-range-thumb{
    width:16px;height:16px;background:linear-gradient(135deg,#0000CD,#1638b7);border-radius:50%;cursor:pointer;border:none;box-shadow:0 2px 6px rgba(0,0,205,.4);
}
.slider-row .slider-val{
    min-width:58px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:500;padding:3px 8px;background:#09090b;border:1px solid #27272a;border-radius:6px;color:#a1a1aa;flex-shrink:0;
}

.app-statusbar{
    background:#18181b;border-top:1px solid #27272a;padding:6px 20px;display:flex;gap:12px;height:34px;align-items:center;font-size:12px;
}
.app-statusbar .sb-section{
    padding:0 12px;flex:1;display:flex;align-items:center;font-family:'JetBrains Mono',monospace;font-size:12px;color:#52525b;overflow:hidden;white-space:nowrap;
}
.app-statusbar .sb-section.sb-fixed{
    flex:0 0 auto;min-width:110px;text-align:center;justify-content:center;padding:3px 12px;background:rgba(0,0,205,.08);border-radius:6px;color:#8da1ff;font-weight:500;
}

.exp-note{padding:10px 20px;font-size:12px;color:#52525b;border-top:1px solid #27272a;text-align:center}
.exp-note a{color:#8da1ff;text-decoration:none}
.exp-note a:hover{text-decoration:underline}

::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#09090b}
::-webkit-scrollbar-thumb{background:#27272a;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#3f3f46}

@media(max-width:980px){
    .app-main-row{flex-direction:column}
    .app-main-right{width:100%}
    .app-main-left{border-right:none;border-bottom:1px solid #27272a}
}
"""

gallery_js = r"""
() => {
function init() {
    if (window.__outpostInitDone) return;

    const dropZone = document.getElementById('media-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick = document.getElementById('upload-click-area');
    const fileInput = document.getElementById('custom-file-input');
    const previewWrap = document.getElementById('single-preview-wrap');
    const previewImg = document.getElementById('single-preview-img');
    const btnUpload = document.getElementById('preview-upload-btn');
    const btnClear = document.getElementById('preview-clear-btn');
    const promptInput = document.getElementById('custom-query-input');
    const runBtnEl = document.getElementById('custom-run-btn');
    const outputArea = document.getElementById('custom-output-textarea');
    const mediaStatus = document.getElementById('sb-media-status');

    if (!dropZone || !fileInput || !promptInput || !previewWrap || !previewImg) {
        setTimeout(init, 250);
        return;
    }

    window.__outpostInitDone = true;
    let mediaState = null;
    let toastTimer = null;
    let examplePoller = null;
    let lastSeenExamplePayload = null;

    function showToast(message, type) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'toast-notification';
            toast.innerHTML = '<span class="toast-icon"></span><span class="toast-text"></span>';
            document.body.appendChild(toast);
        }
        const icon = toast.querySelector('.toast-icon');
        const text = toast.querySelector('.toast-text');
        toast.className = 'toast-notification ' + (type || 'error');
        if (type === 'warning') icon.textContent = '\u26A0';
        else if (type === 'info') icon.textContent = '\u2139';
        else icon.textContent = '\u2717';
        text.textContent = message;
        if (toastTimer) clearTimeout(toastTimer);
        void toast.offsetWidth;
        toast.classList.add('visible');
        toastTimer = setTimeout(() => toast.classList.remove('visible'), 3500);
    }

    function showLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.add('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Processing...';
    }
    function hideLoader() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Done';
    }
    function setRunErrorState() {
        const l = document.getElementById('output-loader');
        if (l) l.classList.remove('active');
        const sb = document.getElementById('sb-run-state');
        if (sb) sb.textContent = 'Error';
    }

    window.__hideLoader = hideLoader;
    window.__setRunErrorState = setRunErrorState;
    window.__showToast = showToast;

    function flashPromptError() {
        promptInput.classList.add('error-flash');
        promptInput.focus();
        setTimeout(() => promptInput.classList.remove('error-flash'), 800);
    }

    function flashOutputError() {
        if (!outputArea) return;
        outputArea.classList.add('error-flash');
        setTimeout(() => outputArea.classList.remove('error-flash'), 800);
    }

    function getValueFromContainer(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return '';
        const el = container.querySelector('textarea, input');
        return el ? (el.value || '') : '';
    }

    function setGradioValue(containerId, value) {
        const container = document.getElementById(containerId);
        if (!container) return false;
        const el = container.querySelector('textarea, input');
        if (!el) return false;
        const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
        const ns = Object.getOwnPropertyDescriptor(proto, 'value');
        if (ns && ns.set) {
            ns.set.call(el, value);
            el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
            el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            return true;
        }
        return false;
    }

    function syncImageToGradio() {
        setGradioValue('hidden-image-b64', mediaState ? mediaState.b64 : '');
        if (mediaStatus) mediaStatus.textContent = mediaState ? '1 image uploaded' : 'No image uploaded';
    }

    function syncPromptToGradio() {
        setGradioValue('prompt-gradio-input', promptInput.value);
    }

    function syncModelToGradio(name) {
        setGradioValue('hidden-model-name', name);
    }

    function renderPreview() {
        if (!mediaState) {
            previewImg.src = '';
            previewImg.style.display = 'none';
            previewWrap.style.display = 'none';
            if (uploadPrompt) uploadPrompt.style.display = 'flex';
            syncImageToGradio();
            return;
        }

        previewWrap.style.display = 'flex';
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        previewImg.src = mediaState.preview || mediaState.b64;
        previewImg.style.display = 'block';
        syncImageToGradio();
    }

    function setPreviewFromFileReader(b64, name) {
        mediaState = {b64, name: name || 'file', mode: 'image'};
        renderPreview();
    }

    function clearPreview() {
        mediaState = null;
        renderPreview();
    }
    window.__clearPreview = clearPreview;

    function processFile(file) {
        if (!file) return;
        if (!file.type.startsWith('image/')) {
            showToast('Only image files are supported', 'error');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => setPreviewFromFileReader(e.target.result, file.name);
        reader.readAsDataURL(file);
    }

    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload) btnUpload.addEventListener('click', () => fileInput.click());
    if (btnClear) btnClear.addEventListener('click', clearPreview);

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
        if (file) processFile(file);
        e.target.value = '';
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
    });

    promptInput.addEventListener('input', syncPromptToGradio);

    function activateModelTab(name) {
        document.querySelectorAll('.model-tab[data-model]').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-model') === name);
        });
        syncModelToGradio(name);
    }

    window.__activateModelTab = activateModelTab;

    document.querySelectorAll('.model-tab[data-model]').forEach(btn => {
        btn.addEventListener('click', () => activateModelTab(btn.getAttribute('data-model')));
    });

    activateModelTab('Qwen3-VL-4B-Instruct');

    function syncSlider(customId, gradioId) {
        const slider = document.getElementById(customId);
        const valSpan = document.getElementById(customId + '-val');
        if (!slider) return;
        slider.addEventListener('input', () => {
            if (valSpan) valSpan.textContent = slider.value;
            const container = document.getElementById(gradioId);
            if (!container) return;
            container.querySelectorAll('input[type="range"],input[type="number"]').forEach(el => {
                const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) {
                    ns.set.call(el, slider.value);
                    el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                }
            });
        });
    }

    syncSlider('custom-max-new-tokens', 'gradio-max-new-tokens');
    syncSlider('custom-temperature', 'gradio-temperature');
    syncSlider('custom-top-p', 'gradio-top-p');
    syncSlider('custom-top-k', 'gradio-top-k');
    syncSlider('custom-repetition-penalty', 'gradio-repetition-penalty');
    syncSlider('custom-gpu-duration', 'gradio-gpu-duration');

    function validateBeforeRun() {
        const promptVal = promptInput.value.trim();
        if (!promptVal) {
            showToast('Please enter your instruction', 'warning');
            flashPromptError();
            return false;
        }
        if (!mediaState) {
            showToast('Please upload an image', 'error');
            return false;
        }
        const currentModel = (document.querySelector('.model-tab.active') || {}).dataset?.model;
        if (!currentModel) {
            showToast('Please select a model', 'error');
            return false;
        }
        return true;
    }

    window.__clickGradioRunBtn = function() {
        if (!validateBeforeRun()) return;
        syncPromptToGradio();
        syncImageToGradio();
        const activeModel = document.querySelector('.model-tab.active');
        if (activeModel) syncModelToGradio(activeModel.getAttribute('data-model'));
        if (outputArea) outputArea.value = '';
        showLoader();
        setTimeout(() => {
            const gradioBtn = document.getElementById('gradio-run-btn');
            if (!gradioBtn) {
                setRunErrorState();
                if (outputArea) outputArea.value = '[ERROR] Run button not found.';
                showToast('Run button not found', 'error');
                return;
            }
            const btn = gradioBtn.querySelector('button');
            if (btn) btn.click(); else gradioBtn.click();
        }, 180);
    };

    if (runBtnEl) runBtnEl.addEventListener('click', () => window.__clickGradioRunBtn());

    const copyBtn = document.getElementById('copy-output-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            try {
                const text = outputArea ? outputArea.value : '';
                if (!text.trim()) {
                    showToast('No output to copy', 'warning');
                    flashOutputError();
                    return;
                }
                await navigator.clipboard.writeText(text);
                showToast('Output copied to clipboard', 'info');
            } catch(e) {
                showToast('Copy failed', 'error');
            }
        });
    }

    const saveBtn = document.getElementById('save-output-btn');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            const text = outputArea ? outputArea.value : '';
            if (!text.trim()) {
                showToast('No output to save', 'warning');
                flashOutputError();
                return;
            }
            const blob = new Blob([text], {type: 'text/plain;charset=utf-8'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'qwen3_vl_outpost_output.txt';
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                URL.revokeObjectURL(a.href);
                document.body.removeChild(a);
            }, 200);
            showToast('Output saved', 'info');
        });
    }

    function applyExamplePayload(raw) {
        try {
            const data = JSON.parse(raw);
            if (data.status !== 'ok') return;

            if (data.model) activateModelTab(data.model);
            if (data.query) {
                promptInput.value = data.query;
                syncPromptToGradio();
            }

            mediaState = {
                b64: data.media || '',
                preview: data.media || '',
                name: data.name || 'example_file',
                mode: 'image'
            };
            renderPreview();

            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
            showToast('Example loaded', 'info');
        } catch (e) {
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
        }
    }

    function startExamplePolling() {
        if (examplePoller) clearInterval(examplePoller);
        let attempts = 0;
        examplePoller = setInterval(() => {
            attempts += 1;
            const current = getValueFromContainer('example-result-data');
            if (current && current !== lastSeenExamplePayload) {
                lastSeenExamplePayload = current;
                clearInterval(examplePoller);
                examplePoller = null;
                applyExamplePayload(current);
                return;
            }
            if (attempts >= 100) {
                clearInterval(examplePoller);
                examplePoller = null;
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Example load timed out', 'error');
            }
        }, 120);
    }

    function triggerExampleLoad(idx) {
        const btnWrap = document.getElementById('example-load-btn');
        const btn = btnWrap ? (btnWrap.querySelector('button') || btnWrap) : null;
        if (!btn) return;

        let attempts = 0;

        function writeIdxAndClick() {
            attempts += 1;

            const ok1 = setGradioValue('example-idx-input', String(idx));
            setGradioValue('example-result-data', '');
            const currentVal = getValueFromContainer('example-idx-input');

            if (ok1 && currentVal === String(idx)) {
                btn.click();
                startExamplePolling();
                return;
            }

            if (attempts < 30) {
                setTimeout(writeIdxAndClick, 100);
            } else {
                document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
                showToast('Failed to initialize example loader', 'error');
            }
        }

        writeIdxAndClick();
    }

    document.querySelectorAll('.example-card[data-idx]').forEach(card => {
        card.addEventListener('click', () => {
            const idx = card.getAttribute('data-idx');
            if (idx === null || idx === undefined || idx === '') return;
            document.querySelectorAll('.example-card.loading').forEach(c => c.classList.remove('loading'));
            card.classList.add('loading');
            showToast('Loading example...', 'info');
            triggerExampleLoad(idx);
        });
    });

    const observerTarget = document.getElementById('example-result-data');
    if (observerTarget) {
        const obs = new MutationObserver(() => {
            const current = getValueFromContainer('example-result-data');
            if (!current || current === lastSeenExamplePayload) return;
            lastSeenExamplePayload = current;
            if (examplePoller) {
                clearInterval(examplePoller);
                examplePoller = null;
            }
            applyExamplePayload(current);
        });
        obs.observe(observerTarget, {childList:true, subtree:true, characterData:true, attributes:true});
    }

    if (outputArea) outputArea.value = '';
    const sb = document.getElementById('sb-run-state');
    if (sb) sb.textContent = 'Ready';
    if (mediaStatus) mediaStatus.textContent = 'No image uploaded';
}
init();
}
"""

wire_outputs_js = r"""
() => {
function watchOutputs() {
    const resultContainer = document.getElementById('gradio-result');
    const outArea = document.getElementById('custom-output-textarea');
    if (!resultContainer || !outArea) { setTimeout(watchOutputs, 500); return; }

    let lastText = '';

    function isErrorText(val) {
        return typeof val === 'string' && val.trim().startsWith('[ERROR]');
    }

    function syncOutput() {
        const el = resultContainer.querySelector('textarea') || resultContainer.querySelector('input');
        if (!el) return;
        const val = el.value || '';
        if (val !== lastText) {
            lastText = val;
            outArea.value = val;
            outArea.scrollTop = outArea.scrollHeight;

            if (val.trim()) {
                if (isErrorText(val)) {
                    if (window.__setRunErrorState) window.__setRunErrorState();
                    if (window.__showToast) window.__showToast('Inference failed', 'error');
                } else {
                    if (window.__hideLoader) window.__hideLoader();
                }
            }
        }
    }

    const observer = new MutationObserver(syncOutput);
    observer.observe(resultContainer, {childList:true, subtree:true, characterData:true, attributes:true});
    setInterval(syncOutput, 500);
}
watchOutputs();
}
"""

FIRE_LOGO_SVG = """
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <path d="M13.5 2.5c.4 2.3-.4 4-1.7 5.6-1.3 1.6-2.6 3-2.6 5 0 1.8 1.2 3.2 2.8 3.2 1.8 0 3.1-1.4 3.1-3.5 0-1.1-.4-2.1-1.2-3.3 2.7 1.2 5.1 4.1 5.1 7.4 0 4-3.1 7.1-7.2 7.1-4.3 0-7.8-3.3-7.8-7.8 0-3.1 1.7-5.5 4-7.8 1.7-1.7 3.9-3.6 4.7-6z" fill="white"/>
</svg>
"""

UPLOAD_PREVIEW_SVG = """
<svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="8" y="14" width="64" height="52" rx="6" fill="none" stroke="#0000CD" stroke-width="2" stroke-dasharray="4 3"/>
    <polygon points="12,62 30,40 42,50 54,34 68,62" fill="rgba(0,0,205,0.15)" stroke="#0000CD" stroke-width="1.5"/>
    <circle cx="28" cy="30" r="6" fill="rgba(0,0,205,0.2)" stroke="#0000CD" stroke-width="1.5"/>
</svg>
"""

COPY_SVG = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M16 1H4C2.9 1 2 1.9 2 3v12h2V3h12V1zm3 4H8C6.9 5 6 5.9 6 7v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>"""
SAVE_SVG = """<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zM7 5h8v4H7V5zm12 14H5v-6h14v6z"/></svg>"""

MODEL_TABS_HTML = "".join([
    f'<button class="model-tab{" active" if m == "Qwen3-VL-4B-Instruct" else ""}" data-model="{m}"><span class="model-tab-label">{m}</span></button>'
    for m in MODEL_CHOICES
])

with gr.Blocks() as demo:
    hidden_image_b64 = gr.Textbox(value="", elem_id="hidden-image-b64", elem_classes="hidden-input", container=False)
    prompt = gr.Textbox(value="", elem_id="prompt-gradio-input", elem_classes="hidden-input", container=False)
    hidden_model_name = gr.Textbox(value="Qwen3-VL-4B-Instruct", elem_id="hidden-model-name", elem_classes="hidden-input", container=False)

    max_new_tokens = gr.Slider(minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS, elem_id="gradio-max-new-tokens", elem_classes="hidden-input", container=False)
    temperature = gr.Slider(minimum=0.1, maximum=4.0, step=0.1, value=0.6, elem_id="gradio-temperature", elem_classes="hidden-input", container=False)
    top_p = gr.Slider(minimum=0.05, maximum=1.0, step=0.05, value=0.9, elem_id="gradio-top-p", elem_classes="hidden-input", container=False)
    top_k = gr.Slider(minimum=1, maximum=1000, step=1, value=50, elem_id="gradio-top-k", elem_classes="hidden-input", container=False)
    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.05, value=1.2, elem_id="gradio-repetition-penalty", elem_classes="hidden-input", container=False)
    gpu_duration_state = gr.Number(value=60, elem_id="gradio-gpu-duration", elem_classes="hidden-input", container=False)

    result = gr.Textbox(value="", elem_id="gradio-result", elem_classes="hidden-input", container=False)

    example_idx = gr.Textbox(value="", elem_id="example-idx-input", elem_classes="hidden-input", container=False)
    example_result = gr.Textbox(value="", elem_id="example-result-data", elem_classes="hidden-input", container=False)
    example_load_btn = gr.Button("Load Example", elem_id="example-load-btn")

    gr.HTML(f"""
    <div class="app-shell">
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">{FIRE_LOGO_SVG}</div>
                <span class="app-title">Qwen3-VL-Outpost</span>
                <span class="app-badge">vision enabled</span>
                <span class="app-badge fast">Image Inference</span>
            </div>
        </div>

        <div class="model-tabs-bar">
            {MODEL_TABS_HTML}
        </div>

        <div class="app-main-row">
            <div class="app-main-left">
                <div id="media-drop-zone">
                    <div id="upload-prompt" class="upload-prompt-modern">
                        <div id="upload-click-area" class="upload-click-area">
                            {UPLOAD_PREVIEW_SVG}
                            <span id="upload-main-text" class="upload-main-text">Click or drag an image here</span>
                            <span id="upload-sub-text" class="upload-sub-text">Upload one image for multimodal inference</span>
                        </div>
                    </div>

                    <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />

                    <div id="single-preview-wrap" class="single-preview-wrap">
                        <div class="single-preview-card">
                            <img id="single-preview-img" src="" alt="Preview" style="display:none;">
                            <div class="preview-overlay-actions">
                                <button id="preview-upload-btn" class="preview-action-btn" title="Replace">Upload</button>
                                <button id="preview-clear-btn" class="preview-action-btn" title="Clear">Clear</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="hint-bar">
                    <b>Mode:</b> Image inference only &nbsp;&middot;&nbsp;
                    <b>Model:</b> Switch between Qwen VL variants &nbsp;&middot;&nbsp;
                    <kbd>Clear</kbd> removes the current image
                </div>

                <div class="examples-section">
                    <div class="examples-title">Quick Examples</div>
                    <div class="examples-scroll">
                        {EXAMPLE_CARDS_HTML}
                    </div>
                </div>
            </div>

            <div class="app-main-right">
                <div class="panel-card">
                    <div id="instruction-title" class="panel-card-title">Vision Instruction</div>
                    <div class="panel-card-body">
                        <label id="query-label" class="modern-label" for="custom-query-input">Query Input</label>
                        <textarea id="custom-query-input" class="modern-textarea" rows="4" placeholder="e.g., perform OCR, solve the problem, describe the image, extract visible text..."></textarea>
                    </div>
                </div>

                <div style="padding:12px 20px;">
                    <button id="custom-run-btn" class="btn-run">
                        <span id="run-btn-label">Run Inference</span>
                    </button>
                </div>

                <div class="output-frame">
                    <div class="out-title">
                        <span id="output-title-label">Raw Output Stream</span>
                        <div class="out-title-right">
                            <button id="copy-output-btn" class="out-action-btn" title="Copy">{COPY_SVG} Copy</button>
                            <button id="save-output-btn" class="out-action-btn" title="Save">{SAVE_SVG} Save File</button>
                        </div>
                    </div>
                    <div class="out-body">
                        <div class="modern-loader" id="output-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Running inference...</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="output-scroll-wrap">
                            <textarea id="custom-output-textarea" class="output-textarea" placeholder="Raw output will appear here..." readonly></textarea>
                        </div>
                    </div>
                </div>

                <div class="settings-group">
                    <div class="settings-group-title">Advanced Settings</div>
                    <div class="settings-group-body">
                        <div class="slider-row">
                            <label>Max new tokens</label>
                            <input type="range" id="custom-max-new-tokens" min="1" max="{MAX_MAX_NEW_TOKENS}" step="1" value="{DEFAULT_MAX_NEW_TOKENS}">
                            <span class="slider-val" id="custom-max-new-tokens-val">{DEFAULT_MAX_NEW_TOKENS}</span>
                        </div>
                        <div class="slider-row">
                            <label>Temperature</label>
                            <input type="range" id="custom-temperature" min="0.1" max="4.0" step="0.1" value="0.6">
                            <span class="slider-val" id="custom-temperature-val">0.6</span>
                        </div>
                        <div class="slider-row">
                            <label>Top-p</label>
                            <input type="range" id="custom-top-p" min="0.05" max="1.0" step="0.05" value="0.9">
                            <span class="slider-val" id="custom-top-p-val">0.9</span>
                        </div>
                        <div class="slider-row">
                            <label>Top-k</label>
                            <input type="range" id="custom-top-k" min="1" max="1000" step="1" value="50">
                            <span class="slider-val" id="custom-top-k-val">50</span>
                        </div>
                        <div class="slider-row">
                            <label>Repetition penalty</label>
                            <input type="range" id="custom-repetition-penalty" min="1.0" max="2.0" step="0.05" value="1.2">
                            <span class="slider-val" id="custom-repetition-penalty-val">1.2</span>
                        </div>
                        <div class="slider-row">
                            <label>GPU Duration (seconds)</label>
                            <input type="range" id="custom-gpu-duration" min="60" max="300" step="30" value="60">
                            <span class="slider-val" id="custom-gpu-duration-val">60</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="exp-note">
            Experimental Qwen VL workspace
        </div>

        <div class="app-statusbar">
            <div class="sb-section" id="sb-media-status">No image uploaded</div>
            <div class="sb-section sb-fixed" id="sb-run-state">Ready</div>
        </div>
    </div>
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=noop, inputs=None, outputs=None, js=gallery_js)
    demo.load(fn=noop, inputs=None, outputs=None, js=wire_outputs_js)

    run_btn.click(
        fn=run_router,
        inputs=[
            hidden_model_name,
            prompt,
            hidden_image_b64,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            gpu_duration_state,
        ],
        outputs=[result],
        js=r"""(model, p, img, mnt, t, tp, tk, rp, gd) => {
            const modelEl = document.querySelector('.model-tab.active');
            const modelVal = modelEl ? modelEl.getAttribute('data-model') : model;
            const promptEl = document.getElementById('custom-query-input');
            const promptVal = promptEl ? promptEl.value : p;

            let imgVal = img;
            const imgContainer = document.getElementById('hidden-image-b64');
            if (imgContainer) {
                const inner = imgContainer.querySelector('textarea, input');
                if (inner) imgVal = inner.value;
            }

            return [modelVal, promptVal, imgVal, mnt, t, tp, tk, rp, gd];
        }""",
    )

    example_load_btn.click(
        fn=load_example_data,
        inputs=[example_idx],
        outputs=[example_result],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(
        css=css,
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["examples"],
    )
