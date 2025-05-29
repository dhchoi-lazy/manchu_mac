import streamlit as st
import torch
import os
import tempfile
import warnings
from PIL import Image
import time
import gc


# Setup cache environment
def setup_cache_environment():
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        local_cache = os.path.join(os.getcwd(), ".hf_cache")
        try:
            os.makedirs(local_cache, exist_ok=True)
            test_file = os.path.join(local_cache, "test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            cache_dir = local_cache
        except (OSError, PermissionError):
            cache_dir = os.path.join(tempfile.gettempdir(), "hf_cache")
            os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

    return cache_dir


cache_directory = setup_cache_environment()

try:
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    import transformers

    def check_transformers_version():
        version = transformers.__version__
        required_version = "4.45.0"

        def version_tuple(v):
            return tuple(map(int, (v.split("."))))

        if version_tuple(version) < version_tuple(required_version):
            st.error(f"❌ Version Error: transformers {version} < {required_version}")
            st.stop()
        return version

    transformers_version = check_transformers_version()

except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.stop()

# Configure torch for MPS compatibility
if torch.backends.mps.is_available():
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    torch.compile = lambda model, *args, **kwargs: model


@st.cache_resource
def load_model(model_name="dhchoi/manchu-llama32-11b-vision-merged"):
    def check_device():
        if torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def optimize_memory():
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    device = check_device()

    with st.spinner(f"Loading Manchu OCR model on {device.upper()}..."):
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_directory,
                local_files_only=False,
                force_download=False,
            )
            processor.tokenizer.padding_side = "left"

            if device == "mps":
                model = MllamaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "16GiB", "cpu": "20GiB"},
                    attn_implementation="eager",
                    cache_dir=cache_directory,
                    local_files_only=False,
                    force_download=False,
                )
            else:
                model = MllamaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_directory,
                    local_files_only=False,
                    force_download=False,
                )

            model.eval()
            optimize_memory()

        except Exception as e:
            st.error(f"❌ Model Loading Error: {str(e)}")
            raise e

    return model, processor


def parse_ocr_response(predicted_text):
    manchu_pred, roman_pred = "", ""
    for line in predicted_text.split("\n"):
        line_stripped = line.strip()
        if line_stripped.lower().startswith("manchu:"):
            manchu_pred = line_stripped.split(":", 1)[-1].strip()
        elif line_stripped.lower().startswith("roman:"):
            roman_pred = line_stripped.split(":", 1)[-1].strip()
    return manchu_pred, roman_pred


def process_image_ocr(model, processor, image, max_length=128):
    try:
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        instruction = (
            "You are an expert OCR system for Manchu script. "
            "Extract the text from the provided image with perfect accuracy. "
            "Format your answer exactly as follows: first line with 'Manchu:' "
            "followed by the Manchu script, then a new line with 'Roman:' "
            "followed by the romanized transliteration."
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": instruction}],
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=input_text,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        first_param_device = next(model.parameters()).device
        inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    use_cache=True,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    num_beams=1,
                )

        generated_text = processor.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        generation_time = time.time() - start_time
        del inputs, outputs

        if generated_text:
            manchu_pred, roman_pred = parse_ocr_response(generated_text)
            return {
                "success": True,
                "manchu": manchu_pred,
                "roman": roman_pred,
                "raw_response": generated_text,
                "generation_time": generation_time,
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate response",
                "generation_time": generation_time,
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "generation_time": 0,
        }


def main():
    st.set_page_config(
        page_title="Manchu OCR",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📜 Manchu Script OCR")
    st.markdown(
        "Upload an image containing Manchu script to extract both the original text and romanized transliteration."
    )

    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
        This application uses a fine-tuned Llama 3.2 Vision model to perform OCR on Manchu script images.
        
        **Features:**
        - Extract Manchu script text
        - Provide romanized transliteration
        - Support for various image formats
        - 🎯 **Demo mode** with sample images
        
        **How to use:**
        1. Check "Use Demo Image" to try with a sample
        2. Or upload your own Manchu script image
        3. Click "Process Image" to run OCR
        
        **Supported formats:**
        - JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP
        """
        )

        st.header("🔧 Model Info")
        device_info = "🖥️ CPU"
        if torch.backends.mps.is_available():
            device_info = "⚡ MPS (Apple Silicon)"

        st.markdown(f"**Device:** {device_info}")
        st.markdown("**Model:** dhchoi/manchu-llama32-11b-vision-merged")
        st.markdown(f"**Transformers:** v{transformers_version}")
        st.markdown(f"**Cache Directory:** `{cache_directory}`")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        demo_option = st.checkbox(
            "🎯 Use Demo Image", help="Try the model with a sample Manchu script image"
        )

        uploaded_file = None
        image = None

        if demo_option:
            demo_path = "samples/validation_sample_0000.jpg"
            if os.path.exists(demo_path):
                image = Image.open(demo_path).convert("RGB")
                st.image(
                    image,
                    caption="Demo Image (validation_sample_0000.jpg)",
                    use_container_width=True,
                )
                st.markdown("**Demo Image:** validation_sample_0000.jpg")
                st.markdown(f"**Size:** {image.size[0]} × {image.size[1]} pixels")

                class MockUploadedFile:
                    def __init__(self, name):
                        self.name = name
                        self.size = os.path.getsize(demo_path)

                uploaded_file = MockUploadedFile("validation_sample_0000.jpg")
            else:
                st.error("Demo image not found. Please upload your own image.")
                demo_option = False

        if not demo_option:
            uploaded_file = st.file_uploader(
                "Choose an image file containing Manchu script",
                type=["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
                help="Upload an image file containing Manchu script text",
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown(f"**Filename:** {uploaded_file.name}")
                st.markdown(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.markdown(f"**File size:** {uploaded_file.size / 1024:.1f} KB")

    with col2:
        st.header("🔍 OCR Results")

        if uploaded_file is not None:
            if st.button("🚀 Process Image", type="primary", use_container_width=True):
                try:
                    model, processor = load_model()

                    with st.spinner("Processing image... This may take a few moments."):
                        if demo_option:
                            pass
                        else:
                            image = Image.open(uploaded_file).convert("RGB")
                        result = process_image_ocr(model, processor, image)

                    if result["success"]:
                        st.success("✅ OCR completed successfully!")

                        st.subheader("📝 Extracted Text")

                        st.markdown("**Manchu Script:**")
                        if result["manchu"]:
                            st.code(result["manchu"], language=None)
                        else:
                            st.warning("No Manchu text detected")

                        st.markdown("**Romanized Transliteration:**")
                        if result["roman"]:
                            st.code(result["roman"], language=None)
                        else:
                            st.warning("No romanized text detected")

                        st.markdown("**Performance:**")
                        st.info(
                            f"⏱️ Processing time: {result['generation_time']:.2f} seconds"
                        )

                        with st.expander("🔍 View Raw Model Response"):
                            st.text(result["raw_response"])

                    else:
                        st.error(
                            f"❌ OCR failed: {result.get('error', 'Unknown error')}"
                        )
                        if result.get("generation_time", 0) > 0:
                            st.info(
                                f"⏱️ Time taken: {result['generation_time']:.2f} seconds"
                            )

                except Exception as e:
                    st.error(f"❌ Failed to load model or process image: {str(e)}")
        else:
            st.info(
                "👆 Please check 'Use Demo Image' or upload an image file to begin OCR processing"
            )

    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and Hugging Face Transformers | Model: [dhchoi/manchu-llama32-11b-vision-merged](https://huggingface.co/dhchoi/manchu-llama32-11b-vision-merged)"
    )


if __name__ == "__main__":
    main()
