"""
Universal evaluation script for GUI grounding tasks.

Supports multiple datasets:
- ScreenSpot (rootsautomation/ScreenSpot)
- UI-Vision (ServiceNow/ui-vision)

Uses modular dataset loaders for easy extension.
"""
import os
import json
import time
import random
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

# Add project root to path for imports and path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set up logging - save to root directory
LOG_PATH = os.path.join(PROJECT_ROOT, 'evaluation.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
# Load .env from project root
env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

from google import genai
from google.genai import types
from google.genai.types import Content, Part
from pydantic import BaseModel
from src.dataset_loaders import get_dataset_loader, BaseDatasetLoader


class GroundingResult(BaseModel):
    """Result for a single grounding evaluation"""
    sample_id: int
    query: str
    ground_truth_bbox: List[float]  # [x1, y1, x2, y2] normalized
    predicted_coord: Optional[Tuple[float, float]] = None  # (x, y) normalized
    point_in_bbox: bool = False
    error: Optional[str] = None
    image_path: Optional[str] = None
    dataset_index: Optional[int] = None


class EvaluationMetrics(BaseModel):
    """Aggregate evaluation metrics"""
    total_samples: int
    successful_predictions: int
    failed_predictions: int
    point_in_bbox_rate: float
    results: List[GroundingResult]


def point_in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
    """Check if a point is inside a bounding box."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def annotate_screenshot(
    image: Image.Image,
    predicted_coord: Optional[Tuple[float, float]],
    ground_truth_bbox: List[float]
) -> Image.Image:
    """Annotate screenshot with predicted coordinate and ground truth bounding box."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    img_width, img_height = image.size

    # Draw ground truth bounding box (green)
    if ground_truth_bbox:
        x1, y1, x2, y2 = ground_truth_bbox
        draw.rectangle(
            (int(x1 * img_width), int(y1 * img_height), 
             int(x2 * img_width), int(y2 * img_height)),
            outline="green",
            width=3
        )
        try:
            font = ImageFont.load_default()
        except ImportError:
            font = None
        text = "Ground Truth"
        text_x = int(x1 * img_width)
        text_y = int(y1 * img_height) - 15
        if text_y < 0:
            text_y = int(y2 * img_height) + 5
        draw.text((text_x, text_y), text, fill="green", font=font)

    # Draw predicted coordinate (red circle with crosshair)
    if predicted_coord:
        x_norm, y_norm = predicted_coord
        x_pixel = int(x_norm * img_width)
        y_pixel = int(y_norm * img_height)

        r, ch = 10, 15
        color = "red"

        draw.ellipse([x_pixel-r, y_pixel-r, x_pixel+r, y_pixel+r], 
                    fill=color, outline="white", width=2)
        draw.line([x_pixel-ch, y_pixel, x_pixel+ch, y_pixel], fill="white", width=2)
        draw.line([x_pixel, y_pixel-ch, x_pixel, y_pixel+ch], fill="white", width=2)

        try:
            font = ImageFont.load_default()
        except ImportError:
            font = None
        text = "Predicted"
        text_x = x_pixel + r + 5
        text_y = y_pixel - 10
        draw.text((text_x, text_y), text, fill="red", font=font)

    return img_copy


def run_gemini_grounding(
    image: Image.Image,
    query: str,
    api_key: Optional[str] = None
) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
    """Run Gemini CUA on a static image to ground a text query."""
    try:
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-computer-use-preview-10-2025"

        # Convert image to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_data = img_bytes.getvalue()

        instruction = f"Click on the UI element described by: '{query}'"

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful web agent that can perform click actions on a web page.\n"
                "Do not ask follow up questions, the user will trust your judgement.\n"
                "You will recieve a screenshot of a web page and a task to complete.\n"
                "You will need to return an action to be performed on the web page and the coordinates of the element to be interacted with.\n"
                "You should return an action that involves clicking on an element and a coordinate.\n"
            ),
            tools=[
                types.Tool(computer_use=types.ComputerUse(environment=types.Environment.ENVIRONMENT_BROWSER))
            ],
        )

        contents = [
            Content(
                role="user",
                parts=[
                    Part(text=instruction),
                    Part.from_bytes(data=img_data, mime_type="image/png")
                ]
            )
        ]

        resp = client.models.generate_content(model=model_name, contents=contents, config=config)

        # Check if response has candidates
        if not resp or not resp.candidates or len(resp.candidates) == 0:
            logger.warning("No candidates in Gemini response")
            return None, "No response from Gemini API"

        cand = resp.candidates[0]
        function_calls = []
        text_parts = []

        # Check if candidate has content
        if not hasattr(cand, 'content') or not cand.content:
            logger.warning("No content in Gemini response candidate")
            return None, "No content in response"

        for part in cand.content.parts:
            if part.function_call:
                function_calls.append(part.function_call)
            elif part.text:
                text_parts.append(part.text)

        response_text = " ".join(text_parts).strip()

        # Extract coordinates from function calls
        action_coords = None
        for fc in function_calls:
            name = fc.name
            args = fc.args if hasattr(fc, 'args') else {}

            if name in ["click_at", "type_text_at", "hover_at", "drag_and_drop", "fill_sensitive_field"]:
                x = args.get("x") if isinstance(args, dict) else getattr(args, 'x', None)
                y = args.get("y") if isinstance(args, dict) else getattr(args, 'y', None)

                if x is not None and y is not None:
                    # Convert from Gemini's 0-999 normalized to 0-1
                    x_norm = x / 1000.0 if x > 1.0 else x
                    y_norm = y / 1000.0 if y > 1.0 else y
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    action_coords = (x_norm, y_norm)
                    break

        if action_coords:
            return action_coords, response_text or f"Action: {name}, Coords: ({action_coords[0]:.3f}, {action_coords[1]:.3f})"

        logger.warning(f"No coordinates found in response.")
        return None, response_text

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error in run_gemini_grounding: {error_msg}")
        return None, error_msg


def get_already_evaluated_samples(output_dir: str) -> set:
    """Get set of dataset indices that have already been evaluated."""
    already_evaluated = set()

    if not os.path.exists(output_dir):
        return already_evaluated

    for status_folder in ["success", "failed"]:
        status_dir = os.path.join(output_dir, status_folder)
        if not os.path.exists(status_dir):
            continue

        try:
            sample_folders = [f for f in os.listdir(status_dir)
                            if os.path.isdir(os.path.join(status_dir, f))]

            for sample_folder in sample_folders:
                metadata_path = os.path.join(status_dir, sample_folder, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            dataset_index = metadata.get("dataset_index")
                            if dataset_index is not None:
                                already_evaluated.add(int(dataset_index))
                    except Exception as e:
                        logger.debug(f"Could not read metadata from {metadata_path}: {e}")
        except Exception as e:
            logger.warning(f"Could not access {status_dir}: {e}")

    logger.info(f"Found {len(already_evaluated)} already evaluated samples.")
    return already_evaluated


def evaluate_grounding(
    dataset_name: str = "ui-vision",
    num_samples: int = 100,
    output_dir: str = "results/evaluation_results",
    api_key: Optional[str] = None,
    random_seed: Optional[int] = None,
    **dataset_kwargs
) -> EvaluationMetrics:
    """
    Main evaluation function for Gemini CUA on grounding datasets.
    
    Args:
        dataset_name: "screenspot" or "ui-vision"
        num_samples: Number of samples to evaluate
        output_dir: Directory to save results
        api_key: Google API key
        random_seed: Random seed for reproducibility
        **dataset_kwargs: Additional arguments for dataset loader
    """
    # Resolve paths relative to project root
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Update dataset paths in kwargs if they're relative
    if dataset_kwargs.get("dataset_path") and not os.path.isabs(dataset_kwargs["dataset_path"]):
        dataset_kwargs["dataset_path"] = os.path.join(PROJECT_ROOT, dataset_kwargs["dataset_path"])
        dataset_kwargs["dataset_path"] = os.path.abspath(dataset_kwargs["dataset_path"])
    
    print(f"Starting evaluation on {num_samples} {dataset_name} samples...")
    logger.info(f"Starting evaluation on {num_samples} {dataset_name} samples")

    os.makedirs(output_dir, exist_ok=True)

    # Get already evaluated samples
    already_evaluated = get_already_evaluated_samples(output_dir)
    if already_evaluated:
        print(f"Found {len(already_evaluated)} already evaluated samples. Will exclude them.")
        logger.info(f"Excluding {len(already_evaluated)} already evaluated samples.")

    # Load dataset using modular loader
    loader = get_dataset_loader(dataset_name, **dataset_kwargs)
    samples = loader.load_samples(
        num_samples=num_samples,
        random_seed=random_seed,
        exclude_indices=already_evaluated
    )

    if not samples:
        print("No new samples to evaluate. Exiting.")
        return EvaluationMetrics(
            total_samples=0, successful_predictions=0, failed_predictions=0,
            point_in_bbox_rate=0.0, results=[]
        )

    results: List[GroundingResult] = []

    # Process each sample
    for idx, sample in enumerate(samples):
        print(f"\nProcessing sample {idx + 1}/{len(samples)}...")
        logger.info(f"Processing sample {idx + 1}/{len(samples)}")

        predicted_coord = None
        response_text = None
        point_in_bbox_result = False
        error_message = None

        try:
            image = loader.get_sample_image(sample)
            query = loader.get_sample_instruction(sample)
            ground_truth_bbox = loader.get_sample_bbox(sample)

            if not ground_truth_bbox:
                error_message = "No ground truth bbox available"
                raise ValueError(error_message)

            logger.info(f"Calling Gemini API with query: '{query}'")
            predicted_coord, response_text = run_gemini_grounding(image, query, api_key)

            if predicted_coord and ground_truth_bbox:
                point_in_bbox_result = point_in_bbox(predicted_coord, ground_truth_bbox)
                logger.info(f"Point in bbox: {point_in_bbox_result}")

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error processing sample {idx}: {error_message}")
            import traceback
            traceback.print_exc()

        sample_id = sample.get("dataset_index", idx)

        result = GroundingResult(
            sample_id=sample_id,
            query=query,
            ground_truth_bbox=ground_truth_bbox or [0.0, 0.0, 1.0, 1.0],
            predicted_coord=predicted_coord,
            point_in_bbox=point_in_bbox_result,
            error=error_message,
            dataset_index=sample_id
        )

        results.append(result)

        # Create folder for this sample
        status_folder = "success" if point_in_bbox_result else "failed"
        sample_dir = os.path.join(output_dir, status_folder, f"sample_{sample_id:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save original image
        try:
            original_image_path = os.path.join(sample_dir, "original.png")
            image.save(original_image_path)
            result.image_path = original_image_path
        except Exception as img_err:
            logger.warning(f"Could not save original image: {img_err}")

        # Save annotated screenshot
        try:
            if image and ground_truth_bbox:
                annotated_image = annotate_screenshot(image, predicted_coord, ground_truth_bbox)
                annotated_image_path = os.path.join(sample_dir, "annotated.png")
                annotated_image.save(annotated_image_path)
        except Exception as annot_err:
            logger.error(f"Error creating annotated image: {annot_err}")

        # Save metadata
        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        time.sleep(0.5)  # Rate limiting

    # Calculate metrics
    successful_predictions = sum(1 for r in results if r.point_in_bbox)
    failed_predictions = len(results) - successful_predictions
    point_in_bbox_rate = successful_predictions / len(results) if results else 0.0

    metrics = EvaluationMetrics(
        total_samples=len(results),
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions,
        point_in_bbox_rate=point_in_bbox_rate,
        results=results
    )

    # Save overall results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(metrics.model_dump(mode="json"), f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {metrics.total_samples}")
    print(f"Successful predictions: {metrics.successful_predictions}")
    print(f"Failed predictions: {metrics.failed_predictions}")
    print(f"Point in bbox rate: {metrics.point_in_bbox_rate:.4f} ({metrics.point_in_bbox_rate*100:.2f}%)")
    print(f"\nResults saved to: {results_path}")
    print("="*60)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Gemini CUA on GUI grounding datasets")
    parser.add_argument("--dataset", type=str, default="ui-vision",
                       choices=["screenspot", "ui-vision", "screenspot-pro"],
                       help="Dataset to evaluate on")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/evaluation_results",
                       help="Output directory (relative to project root)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="Google API key (or use GOOGLE_API_KEY env var)")
    parser.add_argument("--random_seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # UI-Vision specific args
    parser.add_argument("--dataset_path", type=str, default="datasets/ui-vision",
                       help="Path to UI-Vision dataset (relative to project root)")
    parser.add_argument("--task_type", type=str, default="element_grounding",
                       choices=["element_grounding", "layout_grounding"],
                       help="UI-Vision task type")
    parser.add_argument("--subtask", type=str, default="basic",
                       choices=["basic", "functional", "spatial"],
                       help="UI-Vision subtask (for element_grounding)")
    parser.add_argument("--screenspot_pro_path", type=str, default="datasets/ScreenSpot-Pro",
                       help="Path to ScreenSpot-Pro dataset clone (relative to project root)")

    args = parser.parse_args()

    # Prepare dataset kwargs
    dataset_kwargs = {}
    if args.dataset == "ui-vision":
        dataset_kwargs = {
            "dataset_path": args.dataset_path,
            "task_type": args.task_type,
            "subtask": args.subtask
        }
    elif args.dataset in {"screenspot-pro", "screenspot_pro"}:
        dataset_kwargs = {
            "dataset_path": args.screenspot_pro_path
        }

    try:
        metrics = evaluate_grounding(
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            api_key=args.api_key,
            random_seed=args.random_seed,
            **dataset_kwargs
        )
    except Exception as e:
        logger.critical(f"Unhandled error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

