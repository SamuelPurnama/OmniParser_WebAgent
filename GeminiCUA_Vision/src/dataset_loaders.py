"""
Modular dataset loaders for different GUI grounding datasets.
"""
import os
import json
import random
import glob
import logging
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
from datasets import load_dataset


logger = logging.getLogger(__name__)


class BaseDatasetLoader:
    """Base class for dataset loaders."""
    
    def load_samples(
        self,
        num_samples: int = 100,
        random_seed: Optional[int] = None,
        exclude_indices: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """Load samples from the dataset."""
        raise NotImplementedError
    
    def get_sample_image(self, sample: Dict[str, Any]) -> Image.Image:
        """Extract PIL Image from sample."""
        raise NotImplementedError
    
    def get_sample_instruction(self, sample: Dict[str, Any]) -> str:
        """Extract instruction text from sample."""
        raise NotImplementedError
    
    def get_sample_bbox(self, sample: Dict[str, Any]) -> Optional[List[float]]:
        """Extract bounding box [x1, y1, x2, y2] normalized (0-1) from sample."""
        raise NotImplementedError


class ScreenSpotLoader(BaseDatasetLoader):
    """Loader for rootsautomation/ScreenSpot dataset."""
    
    def __init__(self):
        self.dataset_name = "rootsautomation/ScreenSpot"
        self.split = "test"
    
    def load_samples(
        self,
        num_samples: int = 100,
        random_seed: Optional[int] = None,
        exclude_indices: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """Load ScreenSpot dataset from HuggingFace."""
        dataset = load_dataset(self.dataset_name, split=self.split)
        total_size = len(dataset)
        
        exclude_indices = exclude_indices or set()
        available_indices = [i for i in range(total_size) if i not in exclude_indices]
        
        if num_samples > len(available_indices):
            num_samples = len(available_indices)
        
        if random_seed is not None:
            random.seed(random_seed)
        
        sample_indices = random.sample(available_indices, num_samples)
        
        samples = []
        for idx, sample_idx in enumerate(sample_indices):
            sample = dataset[sample_idx]
            samples.append({
                "sample_id": idx,
                "dataset_index": sample_idx,
                "image": sample.get("image"),
                "instruction": sample.get("instruction", ""),
                "bbox": sample.get("bbox"),  # Already normalized
                "data_type": sample.get("data_type", ""),
                "data_source": sample.get("data_source", ""),
                "file_name": sample.get("file_name", ""),
            })
        
        return samples
    
    def get_sample_image(self, sample: Dict[str, Any]) -> Image.Image:
        img = sample.get("image")
        if isinstance(img, Image.Image):
            return img
        elif hasattr(img, 'shape'):  # numpy array
            return Image.fromarray(img)
        else:
            return Image.open(img)
    
    def get_sample_instruction(self, sample: Dict[str, Any]) -> str:
        return sample.get("instruction", "")
    
    def get_sample_bbox(self, sample: Dict[str, Any]) -> Optional[List[float]]:
        bbox = sample.get("bbox")
        if bbox and len(bbox) >= 4:
            return bbox[:4]  # [x1, y1, x2, y2] normalized
        return None


class UIVisionLoader(BaseDatasetLoader):
    """Loader for ServiceNow/ui-vision dataset.
    
    Dataset structure (from cloned repo):
    - ui-vision/
      - annotations/
        - element_grounding/
          - element_grounding_basic.json
          - element_grounding_functional.json
          - element_grounding_spatial.json
        - layout_grounding/
          - layout_grounding.json
      - images/
        - element_grounding/
        - layout_grounding/
    """
    
    def __init__(self, dataset_path: str = "ui-vision", task_type: str = "element_grounding", subtask: str = "basic"):
        """
        Args:
            dataset_path: Path to cloned ui-vision repository
            task_type: "element_grounding" or "layout_grounding"
            subtask: For element_grounding: "basic", "functional", or "spatial"
        """
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.subtask = subtask
        self.annotations_path = os.path.join(dataset_path, "annotations", task_type)
        self.images_path = os.path.join(dataset_path, "images", task_type)
    
    def load_samples(
        self,
        num_samples: int = 100,
        random_seed: Optional[int] = None,
        exclude_indices: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """Load UI-Vision dataset from cloned repository."""
        # Determine annotation file path
        if self.task_type == "element_grounding":
            annotation_file = os.path.join(
                self.annotations_path,
                f"element_grounding_{self.subtask}.json"
            )
        else:  # layout_grounding
            annotation_file = os.path.join(
                self.annotations_path,
                "layout_grounding.json"
            )
        
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file}\n"
                f"Make sure you've cloned the dataset: git clone https://huggingface.co/datasets/ServiceNow/ui-vision"
            )
        
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_path}\n"
                f"Make sure you've cloned the dataset: git clone https://huggingface.co/datasets/ServiceNow/ui-vision"
            )
        
        # Load annotations
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(annotations, dict):
            annotations = list(annotations.values())
        
        total_size = len(annotations)
        
        exclude_indices = exclude_indices or set()
        available_indices = [i for i in range(total_size) if i not in exclude_indices]
        
        if num_samples > len(available_indices):
            num_samples = len(available_indices)
        
        if random_seed is not None:
            random.seed(random_seed)
        
        sample_indices = random.sample(available_indices, num_samples)
        
        samples = []
        for idx, sample_idx in enumerate(sample_indices):
            ann = annotations[sample_idx]
            
            # Extract image path - can be relative or just filename
            img_path_rel = ann.get("image_path") or ann.get("image_filename") or ann.get("img_filename") or ann.get("image")
            if not img_path_rel:
                continue
            
            # Handle relative paths (e.g., "element_grounding/filename.png")
            if "/" in img_path_rel:
                # Extract just the filename
                img_filename = os.path.basename(img_path_rel)
            else:
                img_filename = img_path_rel
            
            # Try to find image in the images directory
            img_path = os.path.join(self.images_path, img_filename)
            if not os.path.exists(img_path):
                # Try with the full relative path
                img_path = os.path.join(self.dataset_path, "images", img_path_rel)
                if not os.path.exists(img_path):
                    continue
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                import logging
                logging.warning(f"Could not load image {img_path}: {e}")
                continue
            
            # Extract bounding box and normalize
            bbox = ann.get("bbox") or ann.get("bounding_box")
            img_size = ann.get("image_size") or ann.get("img_size") or [image.width, image.height]
            
            bbox_normalized = None
            if bbox and len(bbox) >= 4:
                # Normalize bbox to [0, 1]
                x1, y1, x2, y2 = bbox[:4]
                img_width, img_height = img_size[0], img_size[1]
                bbox_normalized = [
                    x1 / img_width,
                    y1 / img_height,
                    x2 / img_width,
                    y2 / img_height
                ]
            
            # Extract instruction
            instruction = ann.get("prompt_to_evaluate") or ann.get("instruction") or ann.get("prompt") or ann.get("query") or ""
            
            samples.append({
                "sample_id": idx,
                "dataset_index": sample_idx,
                "image": image,
                "instruction": instruction,
                "bbox": bbox_normalized,
                "file_name": img_filename,
                "img_size": img_size,
                "task_type": self.task_type,
                "subtask": self.subtask if self.task_type == "element_grounding" else None,
            })
        
        return samples
    
    def get_sample_image(self, sample: Dict[str, Any]) -> Image.Image:
        img = sample.get("image")
        if isinstance(img, Image.Image):
            return img
        elif hasattr(img, 'shape'):  # numpy array
            return Image.fromarray(img)
        else:
            return Image.open(img)
    
    def get_sample_instruction(self, sample: Dict[str, Any]) -> str:
        return sample.get("instruction", "Find the target element")
    
    def get_sample_bbox(self, sample: Dict[str, Any]) -> Optional[List[float]]:
        bbox = sample.get("bbox")
        if bbox and len(bbox) >= 4:
            # Normalize if needed (check if already normalized)
            return bbox[:4]
        return None


class ScreenSpotProLoader(BaseDatasetLoader):
    """Loader for the ScreenSpot-Pro dataset cloned locally with full-resolution images."""

    def __init__(self, dataset_path: str = "ScreenSpot-Pro"):
        self.dataset_path = dataset_path
        self.annotations_dir = os.path.join(dataset_path, "annotations")
        self.images_dir = os.path.join(dataset_path, "images")
        self._cache: Optional[List[Dict[str, Any]]] = None

    def _load_all_samples(self) -> List[Dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self.annotations_dir):
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")

        samples: List[Dict[str, Any]] = []
        annotation_files = sorted(glob.glob(os.path.join(self.annotations_dir, "*.json")))
        if not annotation_files:
            raise ValueError(f"No annotation files found in {self.annotations_dir}")

        for ann_file in annotation_files:
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load annotation file {ann_file}: {e}")
                continue

            for entry in annotations:
                img_rel = entry.get("img_filename") or entry.get("image_path") or entry.get("image")
                if not img_rel:
                    continue
                img_rel = img_rel.replace("\\", "/")
                img_path = os.path.join(self.dataset_path, "images", img_rel)
                if not os.path.exists(img_path):
                    logger.debug(f"Image missing for annotation {entry.get('id')}: {img_path}")
                    continue

                bbox = entry.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue

                img_size = entry.get("img_size")
                if not img_size or len(img_size) < 2:
                    try:
                        with Image.open(img_path) as img_obj:
                            img_size = [img_obj.width, img_obj.height]
                    except Exception as e:
                        logger.warning(f"Could not determine image size for {img_path}: {e}")
                        continue

                width, height = img_size[0], img_size[1]
                if not width or not height:
                    continue

                x1, y1, x2, y2 = bbox[:4]
                bbox_normalized = [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height
                ]

                instruction = entry.get("instruction") or entry.get("instruction_cn") or ""

                samples.append({
                    "image_path": img_path,
                    "instruction": instruction,
                    "bbox": bbox_normalized,
                    "raw_bbox": bbox[:4],
                    "img_size": img_size,
                    "application": entry.get("application"),
                    "platform": entry.get("platform"),
                    "group": entry.get("group"),
                    "ui_type": entry.get("ui_type"),
                    "id": entry.get("id"),
                })

        if not samples:
            raise ValueError("No valid ScreenSpot-Pro samples were loaded.")

        self._cache = samples
        return samples

    def load_samples(
        self,
        num_samples: int = 100,
        random_seed: Optional[int] = None,
        exclude_indices: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        samples = self._load_all_samples()
        total_size = len(samples)

        exclude_indices = exclude_indices or set()
        available_indices = [i for i in range(total_size) if i not in exclude_indices]

        if num_samples > len(available_indices):
            num_samples = len(available_indices)

        if num_samples == 0:
            return []

        if random_seed is not None:
            random.seed(random_seed)

        sample_indices = random.sample(available_indices, num_samples)

        selected: List[Dict[str, Any]] = []
        for idx, sample_idx in enumerate(sample_indices):
            base_sample = samples[sample_idx].copy()
            base_sample["sample_id"] = idx
            base_sample["dataset_index"] = sample_idx
            selected.append(base_sample)

        return selected

    def get_sample_image(self, sample: Dict[str, Any]) -> Image.Image:
        image_path = sample.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def get_sample_instruction(self, sample: Dict[str, Any]) -> str:
        return sample.get("instruction", "")

    def get_sample_bbox(self, sample: Dict[str, Any]) -> Optional[List[float]]:
        return sample.get("bbox")


def get_dataset_loader(dataset_name: str, **kwargs) -> BaseDatasetLoader:
    """Factory function to get the appropriate dataset loader.
    
    Args:
        dataset_name: "screenspot", "ui-vision", or "screenspot-pro"
        **kwargs: Additional arguments for the loader
            - For ui-vision: dataset_path, task_type, subtask
            - For screenspot-pro: dataset_path
    """
    loaders = {
        "screenspot": ScreenSpotLoader,
        "ui-vision": UIVisionLoader,
        "uivision": UIVisionLoader,  # Alias
        "screenspot-pro": ScreenSpotProLoader,
        "screenspotpro": ScreenSpotProLoader,
    }
    
    dataset_name_lower = dataset_name.lower().replace("_", "-")
    
    if dataset_name_lower not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    loader_class = loaders[dataset_name_lower]
    return loader_class(**kwargs)

