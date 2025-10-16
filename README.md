
# **Qwen3-VL-Outpost**

<img width="1756" height="1228" alt="Screenshot 2025-10-16 at 12-17-00 Qwen3-VL-Outpost - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/11114b29-5f43-4a79-8030-7312b343a7a2" />

**Qwen3-VL-Outpost** is a Gradio-based web application for vision-language tasks, leveraging multiple Qwen vision-language models to process images and videos. It provides an intuitive interface for users to input queries, upload media, and generate detailed responses using advanced models like **Qwen3-VL** and **Qwen2.5-VL**.

---

## **Features**

* **Image and Video Inference:** Upload images or videos and input text queries to generate detailed responses.
* **Multiple Model Support:** Choose from the following models:

  * Qwen3-VL-4B-Instruct
  * Qwen3-VL-8B-Instruct
  * Qwen3-VL-4B-Thinking
  * Qwen2.5-VL-3B-Instruct
  * Qwen2.5-VL-7B-Instruct
* **Customizable Parameters:** Adjust advanced settings such as *max new tokens*, *temperature*, *top-p*, *top-k*, and *repetition penalty*.
* **Real-time Streaming:** View model outputs as they are generated.
* **Custom Theme:** Uses a tailored **SteelBlueTheme** for an enhanced user interface.
* **Example Inputs:** Predefined examples for quick testing of image and video inference.

---

## **Installation**

### **Prerequisites**

* Python 3.8 or higher
* Git
* CUDA-compatible GPU (recommended for optimal performance)

---

### **Steps**

#### **1. Clone the Repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-Outpost.git
cd Qwen3-VL-Outpost
```

#### **2. Create a Virtual Environment** *(optional but recommended)*

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### **3. Install Dependencies**

Install the required packages using:

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:

```
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
transformers==4.57.1
huggingface_hub
albumentations
qwen-vl-utils
pyvips-binary
sentencepiece
opencv-python
docling-core
python-docx
torchvision
supervision
matplotlib
pdf2image
num2words
reportlab
html2text
xformers
markdown
requests
pymupdf
loguru
hf_xet
spaces
pyvips
pillow
gradio
einops
httpx
click
torch
fpdf
timm
av
```

---

### **4. Run the Application**

Start the Gradio interface with:

```bash
python app.py
```

This will launch the web interface, accessible via your browser.
The application supports queuing with a maximum size of **50**.

---

## **Usage**

1. **Select a Model:** Choose one of the available Qwen models from the radio buttons.
2. **Upload Media:** Use the image or video upload section to provide input media.
3. **Enter Query:** Input your text query in the provided textbox.
4. **Adjust Settings:** Optionally tweak advanced parameters like *max new tokens* or *temperature* in the accordion.
5. **Submit:** Click the **Submit** button to generate a response.

   * Outputs are displayed in real-time in the **Raw Output Stream** and as formatted Markdown.

---

## **Example Queries**

### **Image Inference**

* “Explain the content in detail.” *(with an uploaded image)*
* “Jsonify Data.” *(for images with tabular data)*

### **Video Inference**

* “Explain the ad in detail.” *(with an uploaded video)*
* “Identify the main actions in the video.”

---

## **Project Structure**

```
Qwen3-VL-Outpost/
│
├── app.py              # Main application script containing the Gradio interface and model logic
├── images/             # Directory for example image files
├── videos/             # Directory for example video files
├── requirements.txt    # List of dependencies required for the project
└── README.md           # Project documentation
```

---

## **Notes**

* The application uses **PyTorch** with GPU acceleration (`torch.cuda`) if available; otherwise, it falls back to CPU.
* Video processing downsamples videos to a maximum of **10 frames** to optimize memory usage.
* Ensure sufficient disk space and memory when loading large models such as **Qwen3-VL-8B-Instruct**.
* The application is designed to run in a browser via Gradio's web interface.

---

## **Contributing**

Contributions are welcome!
To contribute:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit:

   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:

   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---

## **License**

This project is licensed under the **Apache License 2.0**.
See the [LICENSE](LICENSE) file for details.
