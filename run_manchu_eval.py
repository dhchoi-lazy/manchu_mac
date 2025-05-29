import os
import time
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import difflib
from pathlib import Path

from run_manchu_model import OptimizedManchuModel, parse_ocr_response


def calculate_cer(predicted, ground_truth):
    if not ground_truth:
        return 1.0 if predicted else 0.0

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

    return operations / len(gt_chars) if len(gt_chars) > 0 else 0.0


def evaluate_single_sample(model, row, sample_idx, total_samples):
    try:
        start_time = time.time()

        image_path = row["image_path"]
        ground_truth_manchu = row["manchu"]
        ground_truth_roman = row["roman"]
        sample_id = row["id"]

        print(
            f"📸 Processing {sample_idx+1}/{total_samples} (ID: {sample_id}): {os.path.basename(image_path)}"
        )

        if not os.path.exists(image_path):
            print(f"   ❌ Image file not found")
            return {
                "sample_id": sample_id,
                "ground_truth_manchu": ground_truth_manchu,
                "ground_truth_roman": ground_truth_roman,
                "predicted_manchu": "",
                "predicted_roman": "",
                "manchu_exact_match": False,
                "roman_exact_match": False,
                "manchu_cer": 1.0,
                "roman_cer": 1.0,
                "generation_time": 0,
                "success": False,
                "error": "Image file not found",
            }

        result = model.ocr_image(image_path)
        generation_time = time.time() - start_time

        if result.get("success"):
            manchu_pred = result["manchu"]
            roman_pred = result["roman"]

            manchu_exact = manchu_pred.strip() == ground_truth_manchu.strip()
            roman_exact = roman_pred.strip() == ground_truth_roman.strip()

            manchu_cer = calculate_cer(manchu_pred, ground_truth_manchu)
            roman_cer = calculate_cer(roman_pred, ground_truth_roman)

            print(
                f"   GT Manchu: '{ground_truth_manchu}' | Roman: '{ground_truth_roman}'"
            )
            print(f"   PR Manchu: '{manchu_pred}' | Roman: '{roman_pred}'")
            print(
                f"   Accuracy: M={'✓' if manchu_exact else '✗'}(CER:{manchu_cer:.3f}), R={'✓' if roman_exact else '✗'}(CER:{roman_cer:.3f})"
            )
            print(f"   ⏱️  Time: {generation_time:.1f}s")

            if manchu_exact and roman_exact:
                print("   ✅ Perfect match!")

            return {
                "sample_id": sample_id,
                "ground_truth_manchu": ground_truth_manchu,
                "ground_truth_roman": ground_truth_roman,
                "predicted_manchu": manchu_pred,
                "predicted_roman": roman_pred,
                "manchu_exact_match": manchu_exact,
                "roman_exact_match": roman_exact,
                "manchu_cer": manchu_cer,
                "roman_cer": roman_cer,
                "generation_time": generation_time,
                "success": True,
            }
        else:
            print(f"   ❌ OCR failed: {result.get('error', 'Unknown error')}")
            return {
                "sample_id": sample_id,
                "ground_truth_manchu": ground_truth_manchu,
                "ground_truth_roman": ground_truth_roman,
                "predicted_manchu": "",
                "predicted_roman": "",
                "manchu_exact_match": False,
                "roman_exact_match": False,
                "manchu_cer": 1.0,
                "roman_cer": 1.0,
                "generation_time": generation_time,
                "success": False,
                "error": result.get("error", "OCR failed"),
            }

    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {
            "sample_id": row.get("id", sample_idx),
            "ground_truth_manchu": row.get("manchu", ""),
            "ground_truth_roman": row.get("roman", ""),
            "predicted_manchu": "",
            "predicted_roman": "",
            "manchu_exact_match": False,
            "roman_exact_match": False,
            "manchu_cer": 1.0,
            "roman_cer": 1.0,
            "generation_time": 0,
            "success": False,
            "error": str(e),
        }


def calculate_metrics(results):
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

    manchu_correct = sum(1 for r in successful_results if r["manchu_exact_match"])
    roman_correct = sum(1 for r in successful_results if r["roman_exact_match"])

    avg_manchu_cer = sum(r["manchu_cer"] for r in successful_results) / len(
        successful_results
    )
    avg_roman_cer = sum(r["roman_cer"] for r in successful_results) / len(
        successful_results
    )

    generation_times = [r["generation_time"] for r in successful_results]
    avg_generation_time = sum(generation_times) / len(generation_times)

    return {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "success_rate": successful_samples / total_samples,
        "manchu_accuracy": manchu_correct / successful_samples,
        "roman_accuracy": roman_correct / successful_samples,
        "manchu_cer": avg_manchu_cer,
        "roman_cer": avg_roman_cer,
        "average_generation_time": avg_generation_time,
        "total_evaluation_time": sum(generation_times),
    }


def save_results(results, metrics, split):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("evaluation_results", exist_ok=True)

    results_file = f"evaluation_results/results_{split}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "metrics": metrics}, f, ensure_ascii=False, indent=2
        )

    csv_file = f"evaluation_results/results_{split}_{timestamp}.csv"
    pd.DataFrame(results).to_csv(csv_file, index=False, encoding="utf-8")

    print(f"💾 Results saved: {results_file}, {csv_file}")


def print_summary(metrics):
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"📈 Total Samples: {metrics['total_samples']}")
    print(f"✅ Successful: {metrics['successful_samples']}")
    print(f"📊 Success Rate: {metrics['success_rate']:.2%}")
    print()
    print("🔤 MANCHU SCRIPT:")
    print(f"   Accuracy: {metrics['manchu_accuracy']:.2%}")
    print(f"   CER: {metrics['manchu_cer']:.4f}")
    print()
    print("🔤 ROMAN TRANSLITERATION:")
    print(f"   Accuracy: {metrics['roman_accuracy']:.2%}")
    print(f"   CER: {metrics['roman_cer']:.4f}")
    print()
    print("⏱️  PERFORMANCE:")
    print(f"   Average Time: {metrics['average_generation_time']:.2f}s")
    print(f"   Total Time: {metrics['total_evaluation_time']:.1f}s")
    print("=" * 60)


def evaluate_dataset(model, dataset_df, split="validation", max_samples=None):
    if split == "validation":
        data = dataset_df[dataset_df["split"] == "validation"]
    elif split == "train":
        data = dataset_df[dataset_df["split"] == "train"]
    else:
        data = dataset_df

    print(f"📈 Dataset loaded: {len(data)} samples")

    if max_samples:
        data = data.head(max_samples)
        print(f"🔢 Evaluating first {len(data)} samples")

    results = []
    print("🚀 Starting evaluation...")

    for i, (_, row) in enumerate(
        tqdm(data.iterrows(), desc="Evaluating", total=len(data))
    ):
        result = evaluate_single_sample(model, row, i, len(data))
        results.append(result)

        if (i + 1) % 10 == 0:
            current_metrics = calculate_metrics(results)
            if current_metrics.get("successful_samples", 0) > 0:
                print(
                    f"📊 Progress: {i+1}/{len(data)} - Manchu: {current_metrics['manchu_accuracy']:.1%} (CER: {current_metrics['manchu_cer']:.3f}), Roman: {current_metrics['roman_accuracy']:.1%} (CER: {current_metrics['roman_cer']:.3f})"
                )

    final_metrics = calculate_metrics(results)
    save_results(results, final_metrics, split)
    return final_metrics


def main():
    if not Path("manchu_dataset.csv").exists():
        print("❌ Dataset file 'manchu_dataset.csv' not found")
        return

    print("🔄 Loading model...")
    model = OptimizedManchuModel("dhchoi/manchu-llama32-11b-vision-merged")

    print("📊 Loading dataset...")
    dataset_df = pd.read_csv("manchu_dataset.csv")

    metrics = evaluate_dataset(model, dataset_df, split="validation", max_samples=None)

    if metrics:
        print_summary(metrics)


if __name__ == "__main__":
    main()
