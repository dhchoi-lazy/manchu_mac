import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import sys
import os
import time
import glob
import json
from datetime import datetime
import gc

# Configure torch for MPS compatibility
if torch.backends.mps.is_available():
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    torch.compile = lambda model, *args, **kwargs: model

# Global model cache
_cached_model = None
_cached_processor = None
_cached_model_name = None


def check_device():
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available!")
        return "mps"
    else:
        print("⚠️  Using CPU")
        return "cpu"


def optimize_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def get_cached_model(model_name="dhchoi/manchu-llama32-11b-vision-merged"):
    global _cached_model, _cached_processor, _cached_model_name

    if (
        _cached_model is not None
        and _cached_processor is not None
        and _cached_model_name == model_name
    ):
        print("✅ Using cached model")
        return _cached_model, _cached_processor

    print("🔄 Loading model...")
    model, processor = setup_model(model_name)

    _cached_model = model
    _cached_processor = processor
    _cached_model_name = model_name

    return model, processor


def setup_model(model_name="dhchoi/manchu-llama32-11b-vision-merged"):
    device = check_device()
    cache_dir = os.path.join(os.getcwd(), ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
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
            cache_dir=cache_dir,
            local_files_only=False,
            force_download=False,
        )
    else:  # CPU
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            local_files_only=False,
            force_download=False,
        )

    model.eval()
    optimize_memory()
    print("✅ Model loaded successfully")
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


def test_manchu_ocr(model, processor, image_path="image.jpg"):
    if not os.path.exists(image_path):
        return {
            "manchu": "",
            "roman": "",
            "raw_response": "",
            "generation_time": 0,
            "success": False,
            "error": f"Image file not found: {image_path}",
        }

    try:
        image = Image.open(image_path).convert("RGB")
        return process_image_ocr(model, processor, image, max_length=128)
    except Exception as e:
        return {
            "manchu": "",
            "roman": "",
            "raw_response": "",
            "generation_time": 0,
            "success": False,
            "error": f"Failed to load image: {str(e)}",
        }


def get_image_files(path):
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    if os.path.isfile(path):
        if any(path.lower().endswith(ext) for ext in image_extensions):
            return [path]
        else:
            print(f"❌ File {path} is not a supported image format")
            return []

    elif os.path.isdir(path):
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(path, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))

        image_files.sort()
        print(f"📁 Found {len(image_files)} image files in directory: {path}")
        return image_files

    else:
        print(f"❌ Path not found: {path}")
        return []


def save_batch_results(results, output_dir="batch_ocr_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_file = os.path.join(output_dir, f"ocr_results_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_file = os.path.join(output_dir, f"ocr_summary_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("filename,manchu_text,roman_text,generation_time,success\n")
        for result in results:
            filename = os.path.basename(result["image_path"])
            manchu = result.get("manchu", "").replace('"', '""')
            roman = result.get("roman", "").replace('"', '""')
            time_taken = result.get("generation_time", 0)
            success = result.get("success", False)
            f.write(f'"{filename}","{manchu}","{roman}",{time_taken},{success}\n')

    print(f"💾 Results saved: {json_file}, {csv_file}")
    return json_file, csv_file


def batch_ocr_processing(model, processor, image_files):
    results = []
    total_time = 0
    successful_count = 0

    print(f"🔄 Processing {len(image_files)} images...")

    for i, image_path in enumerate(image_files):
        print(f"📸 Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")

        if not os.path.exists(image_path):
            result = {
                "image_path": image_path,
                "success": False,
                "error": "File not found",
                "generation_time": 0,
                "manchu": "",
                "roman": "",
            }
        else:
            result = test_manchu_ocr(model, processor, image_path)
            result["image_path"] = image_path

        if result.get("success"):
            successful_count += 1
            print(f"   ✅ Manchu: {result['manchu']}, Roman: {result['roman']}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")

        total_time += result.get("generation_time", 0)
        results.append(result)

        if (i + 1) % 10 == 0:
            optimize_memory()

    print(
        f"\n📊 Summary: {successful_count}/{len(image_files)} successful ({successful_count/len(image_files)*100:.1f}%)"
    )
    print(
        f"⏱️  Total time: {total_time:.1f}s, Average: {total_time/len(image_files):.2f}s per image"
    )

    save_batch_results(results)
    return results


def interactive_chat(model, processor):
    print("\n🎯 Interactive Manchu OCR")
    print("Enter image file path (or 'quit'/'exit' to end)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nImage path: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            if not os.path.exists(user_input):
                print("❌ Invalid image path")
                continue

            print(f"📸 Processing: {user_input}")
            result = test_manchu_ocr(model, processor, user_input)
            if result.get("success"):
                print(f"Manchu: {result['manchu']}")
                print(f"Roman: {result['roman']}")
                print(f"Time: {result['generation_time']:.2f}s")
            else:
                print(f"❌ Error: {result.get('error')}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        image_files = get_image_files(input_path)

        if not image_files:
            return None

        try:
            model, processor = get_cached_model()

            if len(image_files) == 1:
                result = test_manchu_ocr(model, processor, image_files[0])
                if result.get("success"):
                    print(f"Manchu: {result['manchu']}")
                    print(f"Roman: {result['roman']}")
                    print(f"Time: {result['generation_time']:.2f}s")
                    return result
                else:
                    print(f"❌ Error: {result.get('error')}")
                    return None
            else:
                return batch_ocr_processing(model, processor, image_files)

        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            return None
    else:
        try:
            model, processor = get_cached_model()
            interactive_chat(model, processor)
            return None
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            return None


def process_single_image(
    image_path, model_name="dhchoi/manchu-llama32-11b-vision-merged"
):
    try:
        model, processor = get_cached_model(model_name)
        return test_manchu_ocr(model, processor, image_path)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "manchu": "",
            "roman": "",
            "generation_time": 0,
        }


class OptimizedManchuModel:
    def __init__(self, model_name="dhchoi/manchu-llama32-11b-vision-merged"):
        self.model_name = model_name
        self.model, self.processor = get_cached_model(model_name)

    def ocr_image(self, image_path):
        return test_manchu_ocr(self.model, self.processor, image_path)


if __name__ == "__main__":
    main()
