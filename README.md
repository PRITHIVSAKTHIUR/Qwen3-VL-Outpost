# **Qwen2.5-VL Video Understanding**

A comprehensive multimodal AI application that leverages Qwen2.5-VL models for both image and video understanding tasks. This application provides an intuitive web interface for analyzing visual content using state-of-the-art vision-language models.

---

> [!note]
Demo usage is available now; this link may or may not be available in the future : [https://huggingface.co/spaces/prithivMLmods/Qwen2.5-VL-Outpost](https://huggingface.co/spaces/prithivMLmods/Qwen2.5-VL-Outpost)

---

## Video Understanding

https://github.com/user-attachments/assets/240de7aa-ef0f-46f6-aa81-d8eb863a141b

## Image Inference

![Image Infer](https://github.com/user-attachments/assets/c63444bc-38af-4138-89fb-54b40ca60211)

---

## Features

- **Dual Model Support**: Choose between Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-3B-Instruct models
- **Image Analysis**: Upload and analyze images with natural language queries
- **Video Understanding**: Process videos with intelligent frame sampling and analysis
- **Real-time Streaming**: Get responses as they are generated with streaming output
- **Advanced Configuration**: Fine-tune generation parameters for optimal results
- **Interactive Examples**: Pre-loaded examples for both image and video inference

---

## Models

### Qwen2.5-VL-7B-Instruct
A powerful multimodal AI model developed by Alibaba Cloud that excels at understanding both text and images. This Vision-Language Model (VLM) is designed to handle various visual understanding tasks, including image understanding, video analysis, and multilingual support.

### Qwen2.5-VL-3B-Instruct
An instruction-tuned vision-language model from Alibaba Cloud, built upon the Qwen2-VL series. It excels at understanding and generating text related to both visual and textual inputs, making it capable of tasks like image captioning, visual question answering, object localization, long video understanding, and structured data extraction.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen2.5-VL-Video-Understanding.git
cd Qwen2.5-VL-Video-Understanding
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for optimal performance)
- At least 41GB RAM
- Internet connection for model downloads

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:7860`)

3. Choose between Image Inference or Video Inference tabs

### Image Analysis
- Enter your query in the text box
- Upload an image file
- Select your preferred model
- Adjust advanced parameters if needed
- Click Submit to get analysis

### Video Analysis
- Enter your query describing what you want to analyze
- Upload a video file
- The system will automatically extract 10 evenly spaced frames
- Select your preferred model
- Click Submit for comprehensive video understanding

## Advanced Parameters

- **Max New Tokens**: Control the length of generated responses (1-2048)
- **Temperature**: Adjust creativity vs consistency (0.1-4.0)
- **Top-p**: Nucleus sampling parameter (0.05-1.0)
- **Top-k**: Top-k sampling parameter (1-1000)
- **Repetition Penalty**: Reduce repetitive outputs (1.0-2.0)

## Technical Details

### Video Processing
The application uses intelligent video downsampling to extract 10 representative frames from uploaded videos. Each frame is processed with its timestamp to provide temporal context for analysis.

### GPU Acceleration
The application is optimized for GPU acceleration using CUDA when available. It automatically falls back to CPU processing if no GPU is detected.

### Memory Management
Models are loaded with float16 precision to optimize memory usage while maintaining performance quality.

--- 

## File Structure

```
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── images/               # Example images directory
│   ├── 1.jpg
│   └── 2.jpg
└── videos/               # Example videos directory
    ├── 1.mp4
    ├── 2.mp4
    └── 3.mp4
```

---

## Environment Variables

- `MAX_INPUT_TOKEN_LENGTH`: Maximum input token length (default: 4096)

## Supported Formats

### Images
- JPEG, PNG, BMP, TIFF
- Recommended resolution: Up to 2048x2048 pixels

### Videos
- MP4, AVI, MOV, MKV
- Recommended duration: Up to 60 seconds for optimal processing
- Automatic frame extraction at 10 evenly spaced intervals

## Example Use Cases

### Image Analysis
- Document analysis and data extraction
- Chart and graph interpretation
- Object detection and recognition
- Scene understanding
- OCR and text extraction

### Video Analysis
- Activity recognition
- Scene transition analysis
- Object tracking
- Content summarization
- Advertisement analysis

## Performance Notes

- 7B model provides more detailed and accurate responses but requires more computational resources
- 3B model offers faster processing with good quality results
- GPU acceleration significantly improves response times
- Video processing time scales with video duration and complexity

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce max_new_tokens or use the 3B model
2. **Slow Processing**: Ensure GPU acceleration is available
3. **Model Loading Errors**: Check internet connection for initial model downloads
4. **Video Format Issues**: Convert videos to MP4 format if experiencing problems

### System Requirements
- Minimum 28GB GPU memory for 7B model
- Minimum 16GB GPU memory for 3B model
- 48GB system RAM recommended for optimal performance

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Alibaba Cloud for developing the Qwen2.5-VL models
- Hugging Face for the transformers library
- Gradio for the web interface framework

## Support

For questions and support, please open an issue on the GitHub repository or refer to the official Qwen documentation.

## Citation

If you use this application in your research, please cite the original Qwen2.5-VL papers and this repository.

---
