import torch
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from PIL import Image
import sys
import os
import time
import threading
import glob
import json
from datetime import datetime
import gc


if torch.backends.mps.is_available():
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    torch._dynamo.config.disable = True

    original_compile = torch.compile

    def mps_safe_compile(model, *args, **kwargs):
        print("ℹ️  torch.compile disabled on MPS device")
        return model

    torch.compile = mps_safe_compile


if torch.cuda.is_available():

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_device():
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available!")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA is available!")
        return "cuda"
    else:
        print("⚠️  Using CPU")
        return "cpu"


def optimize_memory():
    """Clear memory and optimize for better performance"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def validate_tensor_inputs(inputs, processor):
    """Validate tensor inputs to prevent CUDA scatter/gather errors"""
    try:

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            vocab_size = processor.tokenizer.vocab_size

            if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                print(
                    f"⚠️  Warning: Found out-of-bounds token IDs. Vocab size: {vocab_size}"
                )

                inputs["input_ids"] = torch.clamp(input_ids, 0, vocab_size - 1)

        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]

            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                print("⚠️  Warning: Non-binary attention mask detected, fixing...")
                inputs["attention_mask"] = (attention_mask > 0).long()

        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if torch.any(torch.isnan(pixel_values)) or torch.any(
                torch.isinf(pixel_values)
            ):
                print(
                    "⚠️  Warning: Invalid pixel values detected, replacing with zeros..."
                )
                inputs["pixel_values"] = torch.nan_to_num(
                    pixel_values, nan=0.0, posinf=1.0, neginf=0.0
                )

        return True

    except Exception as e:
        print(f"⚠️  Tensor validation failed: {e}")
        return False


def display_ocr_result(
    sample_idx, total_samples, image_path, result, ground_truth=None, sample_id=None
):

    if sample_id is not None:
        print(
            f"📸 Processing sample {sample_idx+1}/{total_samples} (ID: {sample_id}): {os.path.basename(image_path)}"
        )
    else:
        print(
            f"📸 Processing image {sample_idx+1}/{total_samples}: {os.path.basename(image_path)}"
        )

    if result and result.get("success", True):
        manchu_pred = result.get("manchu", "")
        roman_pred = result.get("roman", "")
        generation_time = result.get("generation_time", 0)

        if ground_truth:

            gt_manchu = ground_truth.get("manchu", "")
            gt_roman = ground_truth.get("roman", "")

            manchu_exact = manchu_pred.strip() == gt_manchu.strip()
            roman_exact = roman_pred.strip() == gt_roman.strip()

            manchu_cer = result.get("manchu_cer", 0)
            roman_cer = result.get("roman_cer", 0)

            print(f"   GT Manchu: '{gt_manchu}' | Roman: '{gt_roman}'")
            print(f"   PR Manchu: '{manchu_pred}' | Roman: '{roman_pred}'")
            print(
                f"   Accuracy: M={'✓' if manchu_exact else '✗'}(CER:{manchu_cer:.3f}), R={'✓' if roman_exact else '✗'}(CER:{roman_cer:.3f})"
            )
            print(f"⏱️  Time: {generation_time:.1f}s")

            if manchu_exact and roman_exact:
                print("✅ Perfect match!")
            elif not manchu_exact or not roman_exact:
                print("❌ Mismatch detected")
        else:

            print(f"✅ Success! Manchu: {manchu_pred}, Roman: {roman_pred}")
            print(f"⏱️  Time: {generation_time:.1f}s")
    else:

        error_msg = (
            result.get("error", "Unknown error")
            if result
            else "Failed to generate response"
        )
        print(f"❌ {error_msg}")
        if result and "generation_time" in result:
            print(f"⏱️  Time: {result['generation_time']:.1f}s")


def parse_ocr_response(predicted_text):
    manchu_pred, roman_pred = "", ""
    for line in predicted_text.split("\n"):
        line_stripped = line.strip()
        if line_stripped.lower().startswith("manchu:"):
            manchu_pred = line_stripped.split(":", 1)[-1].strip()
        elif line_stripped.lower().startswith("roman:"):
            roman_pred = line_stripped.split(":", 1)[-1].strip()
    return manchu_pred, roman_pred


def calculate_cer(predicted, ground_truth):
    """Calculate Character Error Rate (CER) between predicted and ground truth text"""
    if not ground_truth:
        return 1.0 if predicted else 0.0

    import difflib

    pred_chars = list(predicted.strip())
    gt_chars = list(ground_truth.strip())

    matcher = difflib.SequenceMatcher(None, gt_chars, pred_chars)

    operations = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            operations += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            operations += i2 - i1
        elif tag == "insert":
            operations += j2 - j1

    cer = operations / len(gt_chars) if len(gt_chars) > 0 else 0.0
    return cer


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

    print(f"💾 Batch results saved:")
    print(f"   📄 JSON: {json_file}")
    print(f"   📊 CSV: {csv_file}")

    return json_file, csv_file


def setup_model(model_name="dhchoi/manchu-llama32-11b-vision-merged"):
    device = check_device()

    print("🚀 Optimizing model setup for better performance...")

    cache_dir = os.path.join(os.getcwd(), ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📁 Using cache directory: {cache_dir}")

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )

    processor.tokenizer.padding_side = "left"

    if device == "mps":
        print("⚡ Loading model with MPS optimizations...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "16GiB", "cpu": "20GiB"},
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )
        model.eval()

        print("ℹ️  Skipping torch.compile on MPS device")

    elif device == "cuda":
        print("⚡ Loading model with CUDA optimizations...")
        try:
            model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                use_safetensors=True,
                cache_dir=cache_dir,
            )
            model.eval()

            print("ℹ️  Skipping torch.compile on CUDA to avoid kernel errors")

        except Exception as e:
            print(f"⚠️  CUDA model loading failed: {e}")
            print("🔄 Falling back to CPU...")
            device = "cpu"
            model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            )
            model.eval()

    else:
        print("⚡ Loading model with CPU optimizations...")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
        model.eval()

    print("✅ Model loaded successfully")
    optimize_memory()

    return model, processor


def generate_text_streaming(
    model, processor, prompt, max_length=512, use_ocr_params=False
):
    try:
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        messages = [{"role": "user", "content": prompt}]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        first_param_device = next(model.parameters()).device
        inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

        if use_ocr_params:
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_length,
                "temperature": 0.1,
                "do_sample": True,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True,
            }
        else:
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True,
            }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("Assistant: ", end="", flush=True)
        generated_text = ""
        for token in streamer:
            if token:
                print(token, end="", flush=True)
                generated_text += token

        print()
        thread.join()

        return generated_text.strip()

    except Exception as e:
        return None


def generate_with_image(model, processor, prompt, image_path, max_length=64):
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")

            max_size = 1024
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        else:
            return None

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        try:
            inputs = processor(
                text=input_text,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True,
            )
        except Exception as e:
            print(f"⚠️  Processor failed: {e}")
            return None

        if not validate_tensor_inputs(inputs, processor):
            print("⚠️  Tensor validation failed, skipping this image")
            return None

        first_param_device = next(model.parameters()).device

        try:
            inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
        except Exception as e:
            print(f"⚠️  Failed to move tensors to device: {e}")
            return None

        try:
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
                        output_attentions=False,
                        output_hidden_states=False,
                    )
        except RuntimeError as e:
            if "CUDA" in str(e) or "scatter" in str(e) or "gather" in str(e):
                print(f"⚠️  CUDA kernel error detected: {e}")
                print("🔄 Attempting recovery with CPU fallback...")

                try:
                    model_device = next(model.parameters()).device
                    inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs_cpu,
                            max_new_tokens=max_length,
                            use_cache=False,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.eos_token_id,
                            num_beams=1,
                            output_attentions=False,
                            output_hidden_states=False,
                        )
                    print("✅ CPU fallback successful")
                except Exception as cpu_e:
                    print(f"❌ CPU fallback also failed: {cpu_e}")
                    return None
            else:
                print(f"⚠️  Generation failed: {e}")
                return None

        try:
            generated_text = processor.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        except Exception as e:
            print(f"⚠️  Decoding failed: {e}")
            return None

        del inputs, outputs, image

        if device := next(model.parameters()).device.type:
            if (
                device == "cuda"
                and torch.cuda.memory_allocated()
                > torch.cuda.get_device_properties(0).total_memory * 0.8
            ):
                torch.cuda.empty_cache()
            elif device == "mps":

                pass

        return generated_text.strip()

    except Exception as e:
        print(f"Error in generate_with_image: {e}")
        optimize_memory()
        return None


def test_manchu_ocr(model, processor, image_path="image.jpg"):
    instruction = (
        "You are an expert OCR system for Manchu script. "
        "Extract the text from the provided image with perfect accuracy. "
        "Format your answer exactly as follows: first line with 'Manchu:' "
        "followed by the Manchu script, then a new line with 'Roman:' "
        "followed by the romanized transliteration."
    )

    if not os.path.exists(image_path):
        return None

    start_time = time.time()
    response = generate_with_image(
        model, processor, instruction, image_path, max_length=128
    )
    generation_time = time.time() - start_time

    if response:
        manchu_pred, roman_pred = parse_ocr_response(response)

        return {
            "manchu": manchu_pred,
            "roman": roman_pred,
            "raw_response": response,
            "generation_time": generation_time,
            "success": True,
        }
    else:
        return {
            "manchu": "",
            "roman": "",
            "raw_response": "",
            "generation_time": generation_time,
            "success": False,
        }


def interactive_chat(model, processor):
    print("\nStarting interactive chat session!")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear the conversation history")
    print("Type 'ocr' to test Manchu OCR with a custom image")
    print("-" * 60)

    conversation_history = ""

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation_history = ""
                print("Conversation history cleared!")
                continue

            if user_input.lower() == "ocr":

                image_path = input("Enter image file path: ").strip()

                if not image_path:
                    print("❌ No image path provided.")
                    continue

                if not os.path.exists(image_path):
                    print(f"❌ Image file not found: {image_path}")
                    continue

                image_extensions = {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".webp",
                }
                if not any(
                    image_path.lower().endswith(ext) for ext in image_extensions
                ):
                    print(f"❌ File {image_path} is not a supported image format")
                    continue

                print(f"📸 Processing image: {image_path}")
                result = test_manchu_ocr(model, processor, image_path)
                if result:
                    print(f"Manchu: {result['manchu']}")
                    print(f"Roman: {result['roman']}")
                    print(f"Time: {result['generation_time']:.2f}s")
                else:
                    print("❌ Failed to process the image.")
                continue

            if not user_input:
                continue

            conversation_history += f"Human: {user_input}\nAssistant: "

            start_time = time.time()
            response = generate_text_streaming(
                model,
                processor,
                conversation_history,
                max_length=512,
                use_ocr_params=False,
            )
            generation_time = time.time() - start_time

            if response:
                print(f"({generation_time:.2f}s)")
                conversation_history += f"{response}\n"

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_ocr_processing(model, processor, image_files, save_results=True):
    instruction = (
        "You are an expert OCR system for Manchu script. "
        "Extract the text from the provided image with perfect accuracy. "
        "Format your answer exactly as follows: first line with 'Manchu:' "
        "followed by the Manchu script, then a new line with 'Roman:' "
        "followed by the romanized transliteration."
    )

    results = []
    total_time = 0
    successful_count = 0

    print(f"🔄 Starting batch OCR processing for {len(image_files)} images...")
    print()

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    for i, image_path in enumerate(image_files):
        if not os.path.exists(image_path):
            result = {
                "image_path": image_path,
                "success": False,
                "error": "File not found",
                "generation_time": 0,
                "manchu": "",
                "roman": "",
            }
            display_ocr_result(i, len(image_files), image_path, result)
            results.append(result)
            continue

        try:
            start_time = time.time()

            image = Image.open(image_path).convert("RGB")
            max_size = 1024
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            try:
                inputs = processor(
                    text=input_text,
                    images=image,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                    padding=True,
                )
            except Exception as e:
                result = {
                    "image_path": image_path,
                    "success": False,
                    "error": f"Processor failed: {e}",
                    "generation_time": 0,
                    "manchu": "",
                    "roman": "",
                }
                display_ocr_result(i, len(image_files), image_path, result)
                results.append(result)
                continue

            if not validate_tensor_inputs(inputs, processor):
                result = {
                    "image_path": image_path,
                    "success": False,
                    "error": "Tensor validation failed",
                    "generation_time": 0,
                    "manchu": "",
                    "roman": "",
                }
                display_ocr_result(i, len(image_files), image_path, result)
                results.append(result)
                continue

            first_param_device = next(model.parameters()).device

            try:
                inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
            except Exception as e:
                result = {
                    "image_path": image_path,
                    "success": False,
                    "error": f"Failed to move tensors to device: {e}",
                    "generation_time": 0,
                    "manchu": "",
                    "roman": "",
                }
                display_ocr_result(i, len(image_files), image_path, result)
                results.append(result)
                continue

            try:
                with torch.no_grad():
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            use_cache=True,
                            temperature=0.1,
                            do_sample=True,
                            top_p=0.95,
                            repetition_penalty=1.05,
                            pad_token_id=processor.tokenizer.eos_token_id,
                            num_beams=1,
                            output_attentions=False,
                            output_hidden_states=False,
                        )
            except RuntimeError as e:
                if "CUDA" in str(e) or "scatter" in str(e) or "gather" in str(e):
                    print(
                        f"⚠️  CUDA kernel error detected for {os.path.basename(image_path)}: {e}"
                    )
                    print("🔄 Attempting recovery with CPU fallback...")

                    try:
                        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs_cpu,
                                max_new_tokens=128,
                                use_cache=False,
                                temperature=0.1,
                                do_sample=False,
                                pad_token_id=processor.tokenizer.eos_token_id,
                                num_beams=1,
                                output_attentions=False,
                                output_hidden_states=False,
                            )
                        print(
                            f"✅ CPU fallback successful for {os.path.basename(image_path)}"
                        )
                    except Exception as cpu_e:
                        result = {
                            "image_path": image_path,
                            "success": False,
                            "error": f"CUDA and CPU fallback failed: {cpu_e}",
                            "generation_time": time.time() - start_time,
                            "manchu": "",
                            "roman": "",
                        }
                        display_ocr_result(i, len(image_files), image_path, result)
                        results.append(result)
                        continue
                else:
                    result = {
                        "image_path": image_path,
                        "success": False,
                        "error": f"Generation failed: {e}",
                        "generation_time": time.time() - start_time,
                        "manchu": "",
                        "roman": "",
                    }
                    display_ocr_result(i, len(image_files), image_path, result)
                    results.append(result)
                    continue

            try:
                response = processor.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
            except Exception as e:
                result = {
                    "image_path": image_path,
                    "success": False,
                    "error": f"Decoding failed: {e}",
                    "generation_time": time.time() - start_time,
                    "manchu": "",
                    "roman": "",
                }
                display_ocr_result(i, len(image_files), image_path, result)
                results.append(result)
                continue

            generation_time = time.time() - start_time
            total_time += generation_time

            try:
                del inputs, outputs, image
            except:
                pass

            if response:
                manchu_pred, roman_pred = parse_ocr_response(response)

                result = {
                    "image_path": image_path,
                    "manchu": manchu_pred,
                    "roman": roman_pred,
                    "raw_response": response,
                    "generation_time": generation_time,
                    "success": True,
                }

                successful_count += 1
            else:
                result = {
                    "image_path": image_path,
                    "success": False,
                    "error": "Failed to generate response",
                    "generation_time": generation_time,
                    "manchu": "",
                    "roman": "",
                }

            display_ocr_result(i, len(image_files), image_path, result)
            results.append(result)

            if (i + 1) % 10 == 0:
                optimize_memory()

        except Exception as e:
            result = {
                "image_path": image_path,
                "success": False,
                "error": str(e),
                "generation_time": 0,
                "manchu": "",
                "roman": "",
            }
            display_ocr_result(i, len(image_files), image_path, result)
            results.append(result)

        if (i + 1) % 5 == 0 or (i + 1) == len(image_files):
            avg_time = total_time / (i + 1) if (i + 1) > 0 else 0
            success_rate = (successful_count / (i + 1)) * 100 if (i + 1) > 0 else 0
            print(
                f"📊 Progress: {i+1}/{len(image_files)} - Success: {success_rate:.1f}% - Avg time: {avg_time:.1f}s"
            )
            print()

    print("📊 Batch OCR Summary:")
    print(f"   Total images: {len(image_files)}")
    print(f"   Successful: {successful_count}")
    print(f"   Success rate: {(successful_count/len(image_files)*100):.1f}%")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per image: {(total_time/len(image_files)):.2f}s")
    print()

    if save_results and results:
        save_batch_results(results)

    return results


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        image_files = get_image_files(input_path)

        if not image_files:
            return

        try:
            model, processor = setup_model()

            if len(image_files) == 1:
                test_result = test_manchu_ocr(model, processor, image_files[0])
                if test_result:
                    print(f"Manchu: {test_result['manchu']}")
                    print(f"Roman: {test_result['roman']}")
                    print(f"Time: {test_result['generation_time']:.2f}s")

            else:
                batch_results = batch_ocr_processing(model, processor, image_files)
                successful = sum(1 for r in batch_results if r.get("success", False))
                print(
                    f"Success rate: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)"
                )

        except Exception as e:
            print(f"Failed to initialize model: {e}")

    else:
        try:
            model, processor = setup_model()
            test_result = test_manchu_ocr(model, processor)
            if test_result:
                print(
                    f"OCR test - Manchu: {test_result['manchu']}, Roman: {test_result['roman']}"
                )
            interactive_chat(model, processor)

        except Exception as e:
            print(f"Failed to initialize model: {e}")


class OptimizedManchuModel:
    def __init__(self, model_name="dhchoi/manchu-llama32-11b-vision-merged"):
        self.model_name = model_name
        self.model, self.processor = setup_model(model_name)

    def generate_with_image(self, prompt, image_path, max_length=128):
        return generate_with_image(
            self.model, self.processor, prompt, image_path, max_length
        )


if __name__ == "__main__":
    main()
