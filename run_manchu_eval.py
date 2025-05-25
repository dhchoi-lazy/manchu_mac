import os
import time
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import difflib
from pathlib import Path

# Import our optimized model and shared display function
from run_manchu_model import (
    OptimizedManchuModel,
    parse_ocr_response,
    display_ocr_result,
    calculate_cer,
)


def calculate_accuracy_metrics(predictions, ground_truth):
    exact_matches = sum(
        1 for p, g in zip(predictions, ground_truth) if p.strip() == g.strip()
    )
    total = len(predictions)
    char_correct = 0
    char_total = 0
    total_cer = 0.0

    for pred, gt in zip(predictions, ground_truth):
        pred_clean = pred.strip()
        gt_clean = gt.strip()

        cer = calculate_cer(pred_clean, gt_clean)
        total_cer += cer

        pred_chars = list(pred_clean)
        gt_chars = list(gt_clean)

        max_len = max(len(pred_chars), len(gt_chars))
        char_total += max_len

        for i in range(min(len(pred_chars), len(gt_chars))):
            if pred_chars[i] == gt_chars[i]:
                char_correct += 1

    avg_cer = total_cer / total if total > 0 else 0.0

    return {
        "exact_match_accuracy": exact_matches / total if total > 0 else 0,
        "character_accuracy": char_correct / char_total if char_total > 0 else 0,
        "character_error_rate": avg_cer,
        "total_samples": total,
        "exact_matches": exact_matches,
    }


def load_dataset(dataset_csv):
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    return pd.read_csv(dataset_csv)


def get_dataset_stats(dataset_df):
    train_count = len(dataset_df[dataset_df["split"] == "train"])
    val_count = len(dataset_df[dataset_df["split"] == "validation"])
    return {
        "total_samples": len(dataset_df),
        "train_samples": train_count,
        "validation_samples": val_count,
    }


def evaluate_single_sample(model, ocr_instruction, row, sample_idx, total_samples=None):
    try:
        start_time = time.time()

        image_path = row["image_path"]
        ground_truth_manchu = row["manchu"]
        ground_truth_roman = row["roman"]
        sample_id = row["id"]

        if not os.path.exists(image_path):
            result = {
                "sample_id": sample_id,
                "ground_truth_manchu": ground_truth_manchu,
                "ground_truth_roman": ground_truth_roman,
                "predicted_manchu": "",
                "predicted_roman": "",
                "raw_response": "",
                "manchu_exact_match": False,
                "roman_exact_match": False,
                "manchu_cer": 1.0,
                "roman_cer": 1.0,
                "generation_time": 0,
                "success": False,
                "error": "Image file not found",
                "manchu": "",
                "roman": "",
            }

            # Use shared display function
            ground_truth = {"manchu": ground_truth_manchu, "roman": ground_truth_roman}
            display_ocr_result(
                sample_idx, total_samples, image_path, result, ground_truth, sample_id
            )

            return result

        response = model.generate_with_image(
            ocr_instruction, image_path, max_length=128
        )

        generation_time = time.time() - start_time

        if response:
            manchu_pred, roman_pred = parse_ocr_response(response)

            manchu_exact = manchu_pred.strip() == ground_truth_manchu.strip()
            roman_exact = roman_pred.strip() == ground_truth_roman.strip()

            manchu_cer = calculate_cer(manchu_pred, ground_truth_manchu)
            roman_cer = calculate_cer(roman_pred, ground_truth_roman)

            result = {
                "sample_id": sample_id,
                "ground_truth_manchu": ground_truth_manchu,
                "ground_truth_roman": ground_truth_roman,
                "predicted_manchu": manchu_pred,
                "predicted_roman": roman_pred,
                "raw_response": response,
                "manchu_exact_match": manchu_exact,
                "roman_exact_match": roman_exact,
                "manchu_cer": manchu_cer,
                "roman_cer": roman_cer,
                "generation_time": generation_time,
                "success": True,
                "manchu": manchu_pred,  # For display function compatibility
                "roman": roman_pred,  # For display function compatibility
            }

        else:
            result = {
                "sample_id": sample_id,
                "ground_truth_manchu": ground_truth_manchu,
                "ground_truth_roman": ground_truth_roman,
                "predicted_manchu": "",
                "predicted_roman": "",
                "raw_response": "",
                "manchu_exact_match": False,
                "roman_exact_match": False,
                "manchu_cer": 1.0,
                "roman_cer": 1.0,
                "generation_time": generation_time,
                "success": False,
                "manchu": "",  # For display function compatibility
                "roman": "",  # For display function compatibility
            }

        # Use shared display function
        ground_truth = {"manchu": ground_truth_manchu, "roman": ground_truth_roman}
        display_ocr_result(
            sample_idx, total_samples, image_path, result, ground_truth, sample_id
        )

        return result

    except Exception as e:
        result = {
            "sample_id": row.get("id", sample_idx),
            "ground_truth_manchu": row.get("manchu", ""),
            "ground_truth_roman": row.get("roman", ""),
            "predicted_manchu": "",
            "predicted_roman": "",
            "raw_response": "",
            "manchu_exact_match": False,
            "roman_exact_match": False,
            "manchu_cer": 1.0,
            "roman_cer": 1.0,
            "generation_time": 0,
            "success": False,
            "error": str(e),
            "manchu": "",  # For display function compatibility
            "roman": "",  # For display function compatibility
        }

        # Use shared display function
        ground_truth = {"manchu": row.get("manchu", ""), "roman": row.get("roman", "")}
        display_ocr_result(
            sample_idx, total_samples, image_path, result, ground_truth, sample_id
        )

        return result


def calculate_current_metrics(results):
    if not results:
        return {
            "manchu_acc": 0.0,
            "roman_acc": 0.0,
            "manchu_cer": 1.0,
            "roman_cer": 1.0,
        }

    successful_results = [r for r in results if r["success"]]
    if not successful_results:
        return {
            "manchu_acc": 0.0,
            "roman_acc": 0.0,
            "manchu_cer": 1.0,
            "roman_cer": 1.0,
        }

    manchu_correct = sum(1 for r in successful_results if r["manchu_exact_match"])
    roman_correct = sum(1 for r in successful_results if r["roman_exact_match"])

    avg_manchu_cer = sum(r["manchu_cer"] for r in successful_results) / len(
        successful_results
    )
    avg_roman_cer = sum(r["roman_cer"] for r in successful_results) / len(
        successful_results
    )

    return {
        "manchu_acc": manchu_correct / len(successful_results),
        "roman_acc": roman_correct / len(successful_results),
        "manchu_cer": avg_manchu_cer,
        "roman_cer": avg_roman_cer,
    }


def calculate_final_metrics(results):
    if not results:
        return {}

    successful_results = [r for r in results if r["success"]]
    total_samples = len(results)
    successful_samples = len(successful_results)

    if successful_samples == 0:
        return {
            "total_samples": total_samples,
            "successful_samples": 0,
            "success_rate": 0.0,
            "manchu_accuracy": 0.0,
            "roman_accuracy": 0.0,
            "manchu_cer": 1.0,
            "roman_cer": 1.0,
            "average_generation_time": 0.0,
        }

    # Calculate accuracies
    manchu_correct = sum(1 for r in successful_results if r["manchu_exact_match"])
    roman_correct = sum(1 for r in successful_results if r["roman_exact_match"])

    # Calculate character-level accuracies and CER
    manchu_predictions = [r["predicted_manchu"] for r in successful_results]
    manchu_ground_truth = [r["ground_truth_manchu"] for r in successful_results]
    roman_predictions = [r["predicted_roman"] for r in successful_results]
    roman_ground_truth = [r["ground_truth_roman"] for r in successful_results]

    manchu_metrics = calculate_accuracy_metrics(manchu_predictions, manchu_ground_truth)
    roman_metrics = calculate_accuracy_metrics(roman_predictions, roman_ground_truth)

    # Calculate timing metrics
    generation_times = [r["generation_time"] for r in successful_results]
    avg_generation_time = sum(generation_times) / len(generation_times)

    # Calculate average CER from individual results
    avg_manchu_cer = sum(r["manchu_cer"] for r in successful_results) / len(
        successful_results
    )
    avg_roman_cer = sum(r["roman_cer"] for r in successful_results) / len(
        successful_results
    )

    metrics = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "success_rate": successful_samples / total_samples,
        "manchu_exact_accuracy": manchu_correct / successful_samples,
        "roman_exact_accuracy": roman_correct / successful_samples,
        "manchu_character_accuracy": manchu_metrics["character_accuracy"],
        "roman_character_accuracy": roman_metrics["character_accuracy"],
        "manchu_cer": avg_manchu_cer,
        "roman_cer": avg_roman_cer,
        "average_generation_time": avg_generation_time,
        "total_evaluation_time": sum(generation_times),
        "manchu_metrics": manchu_metrics,
        "roman_metrics": roman_metrics,
    }

    return metrics


def save_evaluation_results(results, metrics, split):
    """Save evaluation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory
    os.makedirs("evaluation_results", exist_ok=True)

    # Save detailed results
    results_file = f"evaluation_results/detailed_results_{split}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save metrics summary
    metrics_file = f"evaluation_results/metrics_summary_{split}_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save CSV for easy analysis
    df = pd.DataFrame(results)
    csv_file = f"evaluation_results/results_{split}_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8")


def print_evaluation_summary(metrics):
    """Print a comprehensive evaluation summary with CER"""
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)

    print(f"📈 Total Samples: {metrics['total_samples']}")
    print(f"✅ Successful: {metrics['successful_samples']}")
    print(f"📊 Success Rate: {metrics['success_rate']:.2%}")
    print()

    print("🔤 MANCHU SCRIPT ACCURACY:")
    print(f"   Exact Match: {metrics['manchu_exact_accuracy']:.2%}")
    print(f"   Character Level: {metrics['manchu_character_accuracy']:.2%}")
    print(f"   Character Error Rate (CER): {metrics['manchu_cer']:.4f}")
    print()

    print("🔤 ROMAN TRANSLITERATION ACCURACY:")
    print(f"   Exact Match: {metrics['roman_exact_accuracy']:.2%}")
    print(f"   Character Level: {metrics['roman_character_accuracy']:.2%}")
    print(f"   Character Error Rate (CER): {metrics['roman_cer']:.4f}")
    print()

    print("⏱️  PERFORMANCE:")
    print(f"   Average Generation Time: {metrics['average_generation_time']:.2f}s")
    print(f"   Total Evaluation Time: {metrics['total_evaluation_time']:.1f}s")
    print("=" * 70)


def evaluate_dataset(
    model,
    ocr_instruction,
    dataset_df,
    split="validation",
    max_samples=None,
    save_results=True,
):
    if split == "train":
        data = dataset_df[dataset_df["split"] == "train"]
    elif split == "validation":
        data = dataset_df[dataset_df["split"] == "validation"]
    else:
        data = dataset_df

    print(f"📈 Dataset loaded: {len(data)} samples")

    if max_samples:
        data = data.head(max_samples)
        print(f"🔢 Evaluating first {len(data)} samples")

    results = []

    print("🚀 Starting evaluation...")
    for i, (_, row) in enumerate(
        tqdm(data.iterrows(), desc="Evaluating samples", total=len(data))
    ):
        result = evaluate_single_sample(model, ocr_instruction, row, i, len(data))
        results.append(result)

        if (i + 1) % 10 == 0:
            current_metrics = calculate_current_metrics(results)
            print(
                f"📊 Progress: {i+1}/{len(data)} - Manchu Acc: {current_metrics['manchu_acc']:.1%} (CER: {current_metrics['manchu_cer']:.3f}), Roman Acc: {current_metrics['roman_acc']:.1%} (CER: {current_metrics['roman_cer']:.3f})"
            )

    final_metrics = calculate_final_metrics(results)

    if save_results:
        save_evaluation_results(results, final_metrics, split)

    return final_metrics


def main():
    if not Path("manchu_dataset.csv").exists():
        return

    model = OptimizedManchuModel("dhchoi/manchu-llama32-11b-vision-merged")
    dataset_df = load_dataset("manchu_dataset.csv")

    ocr_instruction = (
        "You are an expert OCR system for Manchu script. "
        "Extract the text from the provided image with perfect accuracy. "
        "Format your answer exactly as follows: first line with 'Manchu:' "
        "followed by the Manchu script, then a new line with 'Roman:' "
        "followed by the romanized transliteration."
    )

    complete_metrics = evaluate_dataset(
        model, ocr_instruction, dataset_df, split="all", max_samples=None
    )

    if complete_metrics:
        print_evaluation_summary(complete_metrics)


if __name__ == "__main__":
    main()
