# DeepSeek Interactive Chat Application

A Python-based interactive chat application powered by DeepSeek-coder-6.7b-instruct model for code generation and general assistance.

## Features

- 🤖 **Interactive Chat Interface**: Real-time conversation with DeepSeek AI model
- 🧠 **Code Generation**: Specialized in programming tasks and code explanations
- ⚡ **GPU Acceleration**: Optimized for CUDA-enabled GPUs with automatic fallback to CPU
- 🔧 **Flexible Testing**: Quick test mode with predefined prompts
- 💾 **Memory Efficient**: Uses half-precision (float16) to reduce VRAM usage
- 🛡️ **Error Handling**: Robust error handling with graceful degradation

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- Minimum 8GB RAM (16GB+ recommended for GPU usage)
- ~15GB free disk space for model download

### Dependencies
```bash
pip install torch transformers accelerate
```

### GPU Requirements (Optional but Recommended)
- NVIDIA GPU with CUDA support
- Minimum 8GB VRAM (for 6.7B parameter model)

## Installation

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch transformers accelerate
```

3. **Run the application:**
```bash
python 6.7B.py
```

## Usage

### Interactive Chat Mode
When you run the script, you'll be prompted to choose between two modes:

1. **Interactive Chat (Option 1)**: 
   - Engage in real-time conversation with the AI
   - Ask programming questions, request code generation, or general assistance
   - Type `quit`, `exit`, `退出`, or `結束` to end the session

2. **Quick Test (Option 2)**:
   - Runs predefined test prompts to verify model functionality
   - Useful for troubleshooting and performance testing

### Example Usage
```
you: Write a Python function to calculate Fibonacci numbers
DeepSeek: [Generated response with code]

you: Explain recursion in programming
DeepSeek: [Detailed explanation]
```

## Configuration

### Model Settings
The application uses `deepseek-ai/deepseek-coder-6.7b-instruct` by default. You can modify the model in the `load_deepseek_model()` function:

```python
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
```

### Generation Parameters
Adjust these parameters in the `generate_response()` function:
- `max_new_tokens`: Maximum length of generated response (default: 512)
- `temperature`: Controls randomness (default: 0.7)
- `do_sample`: Enable sampling for more diverse outputs

## Performance Monitoring

The application displays:
- GPU memory usage in real-time
- Device allocation information
- Model loading status

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `max_new_tokens` in generation parameters
- Ensure no other GPU-intensive applications are running
- Consider using a smaller model variant

**2. Model Download Fails**
- Check internet connection
- Verify you have sufficient disk space (~15GB)
- Try running with `trust_remote_code=True` (already included)

**3. Slow Performance on CPU**
- CPU inference is significantly slower than GPU
- Consider using a smaller model for CPU-only setups
- Reduce the `max_length` parameter

### GPU Memory Optimization
- Close other applications using GPU memory
- Use `torch.cuda.empty_cache()` if needed
- Monitor memory usage with the built-in display

## Model Information

- **Model**: DeepSeek-coder-6.7b-instruct
- **Parameters**: 6.7 billion
- **Specialization**: Code generation and programming assistance
- **Precision**: FP16 for memory efficiency
- **License**: Check the model's Hugging Face page for licensing terms

## Hardware Recommendations

### Minimum Requirements
- CPU: Any modern processor
- RAM: 8GB
- Storage: 20GB free space

### Recommended Setup
- GPU: NVIDIA RTX 3080/4070 or better (8GB+ VRAM)
- CPU: Multi-core processor (8+ cores)
- RAM: 16GB or more
- Storage: SSD with 25GB+ free space

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your chosen license here]

## Acknowledgments

- [DeepSeek AI](https://github.com/deepseek-ai) for the pre-trained model
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [PyTorch](https://pytorch.org/) for the deep learning framework

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your hardware meets the requirements
3. Ensure all dependencies are correctly installed
4. Open an issue with detailed error information

---

**Note**: First run will download the model (~15GB), which may take some time depending on your internet connection.
