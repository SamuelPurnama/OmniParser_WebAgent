"""
Pipeline to process all images in the ScreenSpot-Pro dataset using OmniParser.
This script segments all images and stores results locally organized by labels.
"""

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import base64
import io
from pathlib import Path

from util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img
)


class ScreenSpotProProcessor:
    """Process ScreenSpot-Pro dataset with OmniParser."""

    def __init__(
        self,
        output_dir="./screenspot_pro_processed",
        yolo_model_path='weights/icon_detect/model.pt',
        caption_model_name="florence2",
        caption_model_path="weights/icon_caption_florence",
        box_threshold=0.05,
        iou_threshold=0.1,
        use_paddleocr=True,
        imgsz=640,
        device='cuda'
    ):
        """Initialize the processor with models and configuration.

        Args:
            output_dir: Directory to save processed results
            yolo_model_path: Path to YOLO model weights
            caption_model_name: Name of caption model ('florence2' or 'blip2')
            caption_model_path: Path to caption model weights
            box_threshold: Box confidence threshold
            iou_threshold: IOU threshold for box filtering
            use_paddleocr: Whether to use PaddleOCR (vs EasyOCR)
            imgsz: Image size for YOLO detection
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processing parameters
        self.box_threshold = box_threshold
        self.iou_threshold = iou_threshold
        self.use_paddleocr = use_paddleocr
        self.imgsz = imgsz
        self.device = device

        print("Loading models...")
        # Load YOLO model
        self.yolo_model = get_yolo_model(model_path=yolo_model_path)
        print(f"✓ YOLO model loaded from {yolo_model_path}")

        # Load caption model
        self.caption_model_processor = get_caption_model_processor(
            model_name=caption_model_name,
            model_name_or_path=caption_model_path,
            device=device
        )
        print(f"✓ Caption model loaded: {caption_model_name}")

    def load_dataset(self, dataset_name="Voxel51/ScreenSpot-Pro", split="test"):
        """Load ScreenSpot-Pro dataset from Hugging Face.

        Args:
            dataset_name: Name of dataset on Hugging Face
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        print(f"\nLoading dataset: {dataset_name} (split: {split})")
        dataset = load_dataset(dataset_name, split=split)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataset

    def process_single_image(self, image, metadata=None):
        """Process a single image with OmniParser.

        Args:
            image: PIL Image object
            metadata: Optional metadata dict for the image

        Returns:
            dict with processed results
        """
        # Calculate box overlay ratio for drawing config
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # Run OCR
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=self.use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt

        # Get labeled image with detected elements
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.yolo_model,
            BOX_TRESHOLD=self.box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            iou_threshold=self.iou_threshold,
            imgsz=self.imgsz
        )

        # Decode the base64 image
        labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))

        # Format parsed content
        parsed_content_formatted = '\n'.join([
            f'Element {i}: {v}' for i, v in enumerate(parsed_content_list)
        ])

        return {
            'labeled_image': labeled_image,
            'label_coordinates': label_coordinates,
            'parsed_content_list': parsed_content_list,
            'parsed_content_formatted': parsed_content_formatted,
            'ocr_text': text,
            'ocr_bbox': ocr_bbox
        }

    def save_results(self, results, sample_idx, metadata=None):
        """Save processed results to disk.

        Args:
            results: Dict containing processed results
            sample_idx: Index of the sample
            metadata: Optional metadata from dataset
        """
        # Create directory structure based on metadata
        if metadata:
            # Organize by group/application/platform
            group = metadata.get('group', 'unknown')
            application = metadata.get('application', 'unknown')
            platform = metadata.get('platform', 'unknown')

            save_dir = self.output_dir / group / application / platform
        else:
            save_dir = self.output_dir / "default"

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save annotated image
        image_path = save_dir / f"sample_{sample_idx:05d}_annotated.png"
        results['labeled_image'].save(image_path)

        # Save parsed content as JSON
        json_path = save_dir / f"sample_{sample_idx:05d}_parsed.json"
        parsed_data = {
            'sample_idx': sample_idx,
            'label_coordinates': results['label_coordinates'],
            'parsed_content': results['parsed_content_list'],
            'ocr_text': results['ocr_text'],
            'metadata': metadata
        }
        with open(json_path, 'w') as f:
            json.dump(parsed_data, f, indent=2)

        # Save parsed content as text
        txt_path = save_dir / f"sample_{sample_idx:05d}_parsed.txt"
        with open(txt_path, 'w') as f:
            f.write(results['parsed_content_formatted'])

        return {
            'image_path': str(image_path),
            'json_path': str(json_path),
            'txt_path': str(txt_path)
        }

    def process_dataset(self, dataset_name="Voxel51/ScreenSpot-Pro", split="test", max_samples=None):
        """Process entire dataset.

        Args:
            dataset_name: Name of dataset on Hugging Face
            split: Dataset split to load
            max_samples: Maximum number of samples to process (None for all)
        """
        # Load dataset
        dataset = self.load_dataset(dataset_name, split)

        # Limit samples if specified
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"Processing first {len(dataset)} samples")

        # Create summary file
        summary_path = self.output_dir / "processing_summary.json"
        summary = {
            'dataset_name': dataset_name,
            'split': split,
            'total_samples': len(dataset),
            'processed_samples': [],
            'failed_samples': []
        }

        # Process each sample
        print(f"\nProcessing {len(dataset)} samples...")
        for idx, sample in enumerate(tqdm(dataset, desc="Processing images")):
            try:
                # Get image from sample
                image = sample['image']

                # Extract metadata
                metadata = {
                    'instruction': sample.get('instruction', ''),
                    'application': sample.get('application', {}).get('label', 'unknown'),
                    'group': sample.get('group', {}).get('label', 'unknown'),
                    'platform': sample.get('platform', {}).get('label', 'unknown')
                }

                # Process image
                results = self.process_single_image(image, metadata)

                # Save results
                saved_paths = self.save_results(results, idx, metadata)

                # Update summary
                summary['processed_samples'].append({
                    'idx': idx,
                    'metadata': metadata,
                    'paths': saved_paths
                })

            except Exception as e:
                print(f"\n✗ Error processing sample {idx}: {str(e)}")
                summary['failed_samples'].append({
                    'idx': idx,
                    'error': str(e)
                })
                continue

        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Processing complete!")
        print(f"  Processed: {len(summary['processed_samples'])} samples")
        print(f"  Failed: {len(summary['failed_samples'])} samples")
        print(f"  Results saved to: {self.output_dir}")
        print(f"  Summary saved to: {summary_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process ScreenSpot-Pro dataset with OmniParser")
    parser.add_argument("--output_dir", type=str, default="./screenspot_pro_processed",
                        help="Output directory for processed results")
    parser.add_argument("--dataset_name", type=str, default="Voxel51/ScreenSpot-Pro",
                        help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to process")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--yolo_model_path", type=str, default="weights/icon_detect/model.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--caption_model", type=str, default="florence2",
                        choices=["florence2", "blip2"],
                        help="Caption model to use")
    parser.add_argument("--caption_model_path", type=str, default="weights/icon_caption_florence",
                        help="Path to caption model weights")
    parser.add_argument("--box_threshold", type=float, default=0.05,
                        help="Box detection threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.1,
                        help="IOU threshold for box filtering")
    parser.add_argument("--use_paddleocr", action="store_true", default=True,
                        help="Use PaddleOCR instead of EasyOCR")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for YOLO detection")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu)")

    args = parser.parse_args()

    # Create processor
    processor = ScreenSpotProProcessor(
        output_dir=args.output_dir,
        yolo_model_path=args.yolo_model_path,
        caption_model_name=args.caption_model,
        caption_model_path=args.caption_model_path,
        box_threshold=args.box_threshold,
        iou_threshold=args.iou_threshold,
        use_paddleocr=args.use_paddleocr,
        imgsz=args.imgsz,
        device=args.device
    )

    # Process dataset
    processor.process_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
