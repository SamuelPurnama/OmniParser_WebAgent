"""
Analysis script for ScreenSpot Pro evaluation results.
Identifies patterns in failures by application, platform, UI type, and other categories.
"""
import os
import sys
import json
import glob
from collections import defaultdict
from typing import Dict, List, Any
from pathlib import Path
import re

# Add project root to path for path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_all_screenspot_pro_annotations(annotations_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load all ScreenSpot Pro annotations and create index by dataset_index."""
    annotation_files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    
    dataset_index = 0
    index_to_annotation = {}
    
    for ann_file in annotation_files:
        app_platform = os.path.basename(ann_file).replace(".json", "")
        parts = app_platform.split("_")
        if len(parts) >= 2:
            application = "_".join(parts[:-1])
            platform = parts[-1]
        else:
            application = app_platform
            platform = "unknown"
        
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            
            for entry in annotations:
                index_to_annotation[dataset_index] = {
                    "application": application,
                    "platform": platform,
                    "ui_type": entry.get("ui_type", "unknown"),
                    "group": entry.get("group", "unknown"),
                    "instruction": entry.get("instruction", ""),
                    "bbox": entry.get("bbox", []),
                    "img_filename": entry.get("img_filename", ""),
                    "annotation_file": app_platform
                }
                dataset_index += 1
        except Exception as e:
            print(f"Error loading {ann_file}: {e}")
            continue
    
    return index_to_annotation

def classify_ui_element(query: str, ui_type: str) -> str:
    """Classify UI element as icon, text, or mixed based on query and ui_type."""
    query_lower = query.lower()
    
    # Check ui_type first
    if ui_type and ui_type.lower() in ["icon", "button"]:
        return "icon"
    if ui_type and ui_type.lower() in ["text", "label"]:
        return "text"
    
    # Infer from query keywords
    icon_keywords = ["tool", "button", "icon", "select", "click", "open", "close", "zoom", "crop", "brush"]
    text_keywords = ["text", "search", "find", "type", "enter", "input", "field", "comment", "label", "keyword"]
    
    icon_count = sum(1 for kw in icon_keywords if kw in query_lower)
    text_count = sum(1 for kw in text_keywords if kw in query_lower)
    
    if icon_count > text_count:
        return "icon"
    elif text_count > icon_count:
        return "text"
    else:
        return "mixed/unknown"

def analyze_query_length(query: str) -> str:
    """Categorize query by length."""
    words = len(query.split())
    if words <= 2:
        return "short (1-2 words)"
    elif words <= 5:
        return "medium (3-5 words)"
    else:
        return "long (6+ words)"

def analyze_bbox_size(bbox: List[float]) -> str:
    """Categorize bounding box by size."""
    if not bbox or len(bbox) < 4:
        return "unknown"
    
    # bbox is normalized [x1, y1, x2, y2]
    width = abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    area = width * height
    
    if area < 0.001:  # Very small (< 0.1% of screen)
        return "very_small"
    elif area < 0.01:  # Small (< 1% of screen)
        return "small"
    elif area < 0.05:  # Medium (< 5% of screen)
        return "medium"
    else:
        return "large"

def load_all_evaluation_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all evaluation results from individual metadata files."""
    all_results = []
    
    for status_folder in ["success", "failed"]:
        status_dir = os.path.join(results_dir, status_folder)
        if not os.path.exists(status_dir):
            continue
        
        sample_folders = [f for f in os.listdir(status_dir)
                         if os.path.isdir(os.path.join(status_dir, f))]
        
        for sample_folder in sample_folders:
            metadata_path = os.path.join(status_dir, sample_folder, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        result = json.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"Warning: Could not read {metadata_path}: {e}")
    
    return all_results

def analyze_evaluation_results(results_dir: str, annotations_dir: str) -> Dict[str, Any]:
    # Resolve paths relative to project root
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(PROJECT_ROOT, results_dir)
    results_dir = os.path.abspath(results_dir)
    
    if not os.path.isabs(annotations_dir):
        annotations_dir = os.path.join(PROJECT_ROOT, annotations_dir)
    annotations_dir = os.path.abspath(annotations_dir)
    """Analyze evaluation results and identify patterns."""
    
    # Load all annotations
    print("Loading ScreenSpot Pro annotations...")
    index_to_annotation = load_all_screenspot_pro_annotations(annotations_dir)
    print(f"Loaded {len(index_to_annotation)} annotations")
    
    # Load all evaluation results from metadata files (aggregates all batches)
    print("Loading all evaluation results from metadata files...")
    results = load_all_evaluation_results(results_dir)
    print(f"Loaded {len(results)} evaluation results from all batches")
    
    total_samples = len(results)
    
    if total_samples == 0:
        raise ValueError(f"No evaluation results found in {results_dir}")
    
    # Initialize counters
    by_application = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    by_platform = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    by_ui_type = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    by_ui_classification = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    by_query_length = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    by_bbox_size = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    by_group = defaultdict(lambda: {"success": 0, "fail": 0, "no_coord": 0})
    
    # Analyze each result
    for result in results:
        dataset_index = result.get("dataset_index")
        if dataset_index is None:
            continue
        
        annotation = index_to_annotation.get(dataset_index, {})
        query = result.get("query", "")
        predicted_coord = result.get("predicted_coord")
        point_in_bbox = result.get("point_in_bbox", False)
        bbox = result.get("ground_truth_bbox", [])
        
        # Determine status
        if predicted_coord is None:
            status = "no_coord"
        elif point_in_bbox:
            status = "success"
        else:
            status = "fail"
        
        # Count by application
        app = annotation.get("application", "unknown")
        by_application[app][status] += 1
        
        # Count by platform
        platform = annotation.get("platform", "unknown")
        by_platform[platform][status] += 1
        
        # Count by UI type
        ui_type = annotation.get("ui_type", "unknown")
        by_ui_type[ui_type][status] += 1
        
        # Count by UI classification (icon vs text)
        ui_class = classify_ui_element(query, ui_type)
        by_ui_classification[ui_class][status] += 1
        
        # Count by query length
        query_len = analyze_query_length(query)
        by_query_length[query_len][status] += 1
        
        # Count by bbox size
        bbox_size = analyze_bbox_size(bbox)
        by_bbox_size[bbox_size][status] += 1
        
        # Count by group
        group = annotation.get("group", "unknown")
        by_group[group][status] += 1
    
    # Calculate success rates
    def calc_success_rate(counts: Dict[str, int]) -> float:
        total = counts.get("success", 0) + counts.get("fail", 0) + counts.get("no_coord", 0)
        if total == 0:
            return 0.0
        return counts.get("success", 0) / total
    
    # Calculate overall success rate
    successful_count = sum(1 for r in results if r.get("point_in_bbox", False))
    overall_success_rate = successful_count / total_samples if total_samples > 0 else 0.0
    
    # Build analysis report
    analysis = {
        "total_samples": total_samples,
        "overall_success_rate": overall_success_rate,
        "by_application": {},
        "by_platform": {},
        "by_ui_type": {},
        "by_ui_classification": {},
        "by_query_length": {},
        "by_bbox_size": {},
        "by_group": {}
    }
    
    # Sort and format results
    for app, counts in sorted(by_application.items()):
        analysis["by_application"][app] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    for platform, counts in sorted(by_platform.items()):
        analysis["by_platform"][platform] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    for ui_type, counts in sorted(by_ui_type.items()):
        analysis["by_ui_type"][ui_type] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    for ui_class, counts in sorted(by_ui_classification.items()):
        analysis["by_ui_classification"][ui_class] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    for qlen, counts in sorted(by_query_length.items()):
        analysis["by_query_length"][qlen] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    for bbox_size, counts in sorted(by_bbox_size.items()):
        analysis["by_bbox_size"][bbox_size] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    for group, counts in sorted(by_group.items()):
        analysis["by_group"][group] = {
            **counts,
            "total": sum(counts.values()),
            "success_rate": calc_success_rate(counts)
        }
    
    return analysis

def generate_report(analysis: Dict[str, Any], output_file: str):
    """Generate a human-readable analysis report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("SCREENSPOT PRO EVALUATION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total Samples Evaluated: {analysis['total_samples']}")
    lines.append(f"Overall Success Rate: {analysis['overall_success_rate']:.2%}")
    lines.append("")
    
    # By Application
    lines.append("=" * 80)
    lines.append("BY APPLICATION")
    lines.append("=" * 80)
    lines.append(f"{'Application':<30} {'Success':<10} {'Fail':<10} {'No Coord':<12} {'Total':<10} {'Success Rate':<15}")
    lines.append("-" * 80)
    
    app_data = sorted(
        analysis["by_application"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )
    
    for app, data in app_data:
        lines.append(
            f"{app:<30} {data['success']:<10} {data['fail']:<10} "
            f"{data['no_coord']:<12} {data['total']:<10} {data['success_rate']:<15.2%}"
        )
    lines.append("")
    
    # By Platform
    lines.append("=" * 80)
    lines.append("BY PLATFORM")
    lines.append("=" * 80)
    lines.append(f"{'Platform':<20} {'Success':<10} {'Fail':<10} {'No Coord':<12} {'Total':<10} {'Success Rate':<15}")
    lines.append("-" * 80)
    
    platform_data = sorted(
        analysis["by_platform"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )
    
    for platform, data in platform_data:
        lines.append(
            f"{platform:<20} {data['success']:<10} {data['fail']:<10} "
            f"{data['no_coord']:<12} {data['total']:<10} {data['success_rate']:<15.2%}"
        )
    lines.append("")
    
    # By UI Classification (Icon vs Text)
    lines.append("=" * 80)
    lines.append("BY UI ELEMENT TYPE (Icon vs Text)")
    lines.append("=" * 80)
    lines.append(f"{'UI Type':<20} {'Success':<10} {'Fail':<10} {'No Coord':<12} {'Total':<10} {'Success Rate':<15}")
    lines.append("-" * 80)
    
    ui_class_data = sorted(
        analysis["by_ui_classification"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )
    
    for ui_class, data in ui_class_data:
        lines.append(
            f"{ui_class:<20} {data['success']:<10} {data['fail']:<10} "
            f"{data['no_coord']:<12} {data['total']:<10} {data['success_rate']:<15.2%}"
        )
    lines.append("")
    
    # By UI Type (from annotations)
    lines.append("=" * 80)
    lines.append("BY UI TYPE (from annotations)")
    lines.append("=" * 80)
    lines.append(f"{'UI Type':<30} {'Success':<10} {'Fail':<10} {'No Coord':<12} {'Total':<10} {'Success Rate':<15}")
    lines.append("-" * 80)
    
    ui_type_data = sorted(
        analysis["by_ui_type"].items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )
    
    for ui_type, data in ui_type_data[:10]:  # Top 10
        lines.append(
            f"{ui_type:<30} {data['success']:<10} {data['fail']:<10} "
            f"{data['no_coord']:<12} {data['total']:<10} {data['success_rate']:<15.2%}"
        )
    lines.append("")
    
    # By Query Length
    lines.append("=" * 80)
    lines.append("BY QUERY LENGTH")
    lines.append("=" * 80)
    lines.append(f"{'Query Length':<20} {'Success':<10} {'Fail':<10} {'No Coord':<12} {'Total':<10} {'Success Rate':<15}")
    lines.append("-" * 80)
    
    query_len_data = sorted(
        analysis["by_query_length"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )
    
    for qlen, data in query_len_data:
        lines.append(
            f"{qlen:<20} {data['success']:<10} {data['fail']:<10} "
            f"{data['no_coord']:<12} {data['total']:<10} {data['success_rate']:<15.2%}"
        )
    lines.append("")
    
    # By BBox Size
    lines.append("=" * 80)
    lines.append("BY BOUNDING BOX SIZE")
    lines.append("=" * 80)
    lines.append(f"{'BBox Size':<20} {'Success':<10} {'Fail':<10} {'No Coord':<12} {'Total':<10} {'Success Rate':<15}")
    lines.append("-" * 80)
    
    bbox_size_data = sorted(
        analysis["by_bbox_size"].items(),
        key=lambda x: x[1]["success_rate"],
        reverse=True
    )
    
    for bbox_size, data in bbox_size_data:
        lines.append(
            f"{bbox_size:<20} {data['success']:<10} {data['fail']:<10} "
            f"{data['no_coord']:<12} {data['total']:<10} {data['success_rate']:<15.2%}"
        )
    lines.append("")
    
    # Key Findings
    lines.append("=" * 80)
    lines.append("KEY FINDINGS")
    lines.append("=" * 80)
    lines.append("")
    
    # Best and worst performing applications
    app_rates = [(app, data["success_rate"]) for app, data in app_data if data["total"] >= 3]
    if app_rates:
        best_app = max(app_rates, key=lambda x: x[1])
        worst_app = min(app_rates, key=lambda x: x[1])
        lines.append(f"Best Performing Application: {best_app[0]} ({best_app[1]:.2%} success rate)")
        lines.append(f"Worst Performing Application: {worst_app[0]} ({worst_app[1]:.2%} success rate)")
        lines.append("")
    
    # Icon vs Text
    icon_data = analysis["by_ui_classification"].get("icon", {})
    text_data = analysis["by_ui_classification"].get("text", {})
    if icon_data.get("total", 0) > 0 and text_data.get("total", 0) > 0:
        icon_rate = icon_data.get("success_rate", 0)
        text_rate = text_data.get("success_rate", 0)
        lines.append(f"Icon Elements Success Rate: {icon_rate:.2%} ({icon_data.get('total', 0)} samples)")
        lines.append(f"Text Elements Success Rate: {text_rate:.2%} ({text_data.get('total', 0)} samples)")
        if icon_rate > text_rate:
            lines.append("→ Icons perform better than text elements")
        elif text_rate > icon_rate:
            lines.append("→ Text elements perform better than icons")
        else:
            lines.append("→ Similar performance between icons and text")
        lines.append("")
    
    # BBox size findings
    bbox_rates = [(size, data["success_rate"]) for size, data in bbox_size_data if data["total"] >= 5]
    if bbox_rates:
        best_size = max(bbox_rates, key=lambda x: x[1])
        worst_size = min(bbox_rates, key=lambda x: x[1])
        lines.append(f"Best Performing BBox Size: {best_size[0]} ({best_size[1]:.2%} success rate)")
        lines.append(f"Worst Performing BBox Size: {worst_size[0]} ({worst_size[1]:.2%} success rate)")
        lines.append("")
    
    # No coordinate returns
    total_no_coord = sum(data.get("no_coord", 0) for data in analysis["by_application"].values())
    if total_no_coord > 0:
        lines.append(f"Samples with No Coordinates Returned: {total_no_coord} ({total_no_coord/analysis['total_samples']:.2%})")
        lines.append("→ These are cases where Gemini didn't return any click coordinates")
        lines.append("")
    
    lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(lines)
    
    with open(output_file, "w") as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ScreenSpot Pro evaluation results")
    parser.add_argument("--results_dir", type=str, default="results/screenspot_pro_evaluation_results",
                       help="Directory containing evaluation results (relative to project root)")
    parser.add_argument("--annotations_dir", type=str, default="datasets/ScreenSpot-Pro/annotations",
                       help="Directory containing ScreenSpot Pro annotations (relative to project root)")
    parser.add_argument("--output", type=str, default="results/evaluation_analysis_report.txt",
                       help="Output file for the analysis report (relative to project root)")
    
    args = parser.parse_args()
    
    # Resolve output path relative to project root
    if not os.path.isabs(args.output):
        args.output = os.path.join(PROJECT_ROOT, args.output)
    args.output = os.path.abspath(args.output)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing evaluation results from: {args.results_dir}")
    print(f"Using annotations from: {args.annotations_dir}")
    
    analysis = analyze_evaluation_results(args.results_dir, args.annotations_dir)
    
    # Save JSON analysis
    json_output = args.output.replace(".txt", ".json")
    with open(json_output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nJSON analysis saved to: {json_output}")
    
    # Generate and save report
    report = generate_report(analysis, args.output)
    print(f"\nAnalysis report saved to: {args.output}")

