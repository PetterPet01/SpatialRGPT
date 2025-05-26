# File: generalized_sg_generator_hf.py

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import time
import json
import re
import warnings
from mmengine import Config
import gc
import os
import shutil
import time
from datetime import datetime
from math import pi

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image

from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV, Fisheye624, Pinhole, Spherical
import open3d as o3d

import open3d as o3d # Should already be there from unik3d related imports
from wis3d import Wis3D
import matplotlib # For color_by_instance
from scipy.spatial.transform import Rotation # For oriented_bbox_to_center_euler_extent
from collections import Counter # For pcd_denoise_dbscan

# OSDSUTILS imports (ensure osdsynth is in PYTHONPATH or installed)
try:
    from osdsynth.processor.captions import CaptionImage
    from osdsynth.processor.pointcloud import PointCloudReconstruction
    from osdsynth.processor.prompt import PromptGenerator as QAPromptGenerator
    from osdsynth.processor.instruction import PromptGenerator as FactPromptGenerator
    from osdsynth.processor.segment import SegmentImage
    from osdsynth.utils.logger import SkipImageException, setup_logger
    from osdsynth.processor.wrappers.sam import (
        convert_detections_to_dict, convert_detections_to_list,
        crop_detections_with_xyxy, filter_detections,
        get_sam_segmentation_from_xyxy, mask_subtract_contained,
        post_process_mask, sort_detections_by_area
    )
    from osdsynth.processor.wrappers.ram import run_tagging_model
    #from osdsynth.processor.wrappers.unik3d_demo import get_depth_model as get_unik3d_depth_model
    OSDSYNTH_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import osdsynth components: {e}. Ensure osdsynth is correctly installed and in PYTHONPATH.")
    OSDSYNTH_AVAILABLE = False
    # Add dummy classes if osdsynth is not available to allow script to be parsed
    class CaptionImage: pass
    class PointCloudReconstruction: pass
    class QAPromptGenerator: pass
    class FactPromptGenerator: pass
    class SegmentImage: pass
    class SkipImageException(Exception): pass
    def setup_logger(name="dummy"): 
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers: # Avoid duplicate handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    # Dummy functions for wrappers if needed for parsing
    def convert_detections_to_dict(*args, **kwargs): return {}
    def convert_detections_to_list(*args, **kwargs): return []
    def get_unik3d_depth_model(*args, **kwargs): return None


# Hugging Face Transformers imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    warnings.warn("Hugging Face Transformers not found. LLM rephrasing will not be available.")

# --- Helper functions for Point Cloud Processing and Visualization (Derived from pointcloud.py logic) ---

def process_pcd_for_unik3d(cfg, pcd, run_dbscan=True):
    """Process PointCloud: Denoise and Downsample for UniK3D output."""
    if not pcd.has_points() or len(pcd.points) == 0:
        return pcd

    # Statistical outlier removal
    # Scale for voxel_down_sample can be tricky. If points are already metric,
    # fixed values might be better than scale-dependent ones if scale varies wildly.
    # For now, let's use a small fixed voxel size for downsampling.
    try:
        # std_ratio default in O3D is 2.0. 1.2 is more aggressive.
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=cfg.get("pcd_sor_neighbors", 20), 
                                                std_ratio=cfg.get("pcd_sor_std_ratio", 1.5))
    except RuntimeError as e:
        # print(f"Warning: Statistical outlier removal failed: {e}")
        pass # Continue with the original pcd if SOR fails

    if not pcd.has_points() or len(pcd.points) == 0:
        return pcd

    # Voxel down-sampling
    voxel_size = cfg.get("pcd_voxel_size", 0.01) # e.g., 1cm
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if not pcd.has_points() or len(pcd.points) == 0:
        return pcd

    if cfg.get("dbscan_remove_noise", True) and run_dbscan:
        pcd = pcd_denoise_dbscan_for_unik3d(
            pcd,
            eps=cfg.get("dbscan_eps", 0.05), # eps might need adjustment based on point density
            min_points=cfg.get("dbscan_min_points", 10)
        )
    return pcd

def pcd_denoise_dbscan_for_unik3d(pcd: o3d.geometry.PointCloud, eps=0.05, min_points=10) -> o3d.geometry.PointCloud:
    """Denoise PointCloud using DBSCAN for UniK3D output."""
    if not pcd.has_points() or len(pcd.points) < min_points:
        return pcd

    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except RuntimeError as e:
        # print(f"Warning: DBSCAN clustering failed: {e}")
        return pcd # Return original pcd if clustering fails

    counts = Counter(labels)
    if -1 in counts:
        del counts[-1]  # Remove noise label

    if not counts:  # No clusters found
        return o3d.geometry.PointCloud() # Return empty PCD

    # Find the largest cluster
    largest_cluster_label = counts.most_common(1)[0][0]
    
    # Select points belonging to the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    
    if len(largest_cluster_indices) < min_points: # If largest cluster is too small
        return o3d.geometry.PointCloud() # Return empty PCD

    return pcd.select_by_index(largest_cluster_indices)


def get_bounding_box_for_unik3d(cfg, pcd):
    """Get Axis-Aligned and Oriented Bounding Box for UniK3D output."""
    if not pcd.has_points() or len(pcd.points) < 3: # Need at least 3 points for OBB
        # Return empty/default BBoxes if not enough points
        aabb = o3d.geometry.AxisAlignedBoundingBox()
        obb = o3d.geometry.OrientedBoundingBox()
        return aabb, obb

    axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()
    try:
        # Robust=True can help with near-degenerate cases but might be slower
        oriented_bbox = pcd.get_oriented_bounding_box(robust=cfg.get("obb_robust", True))
    except RuntimeError:
        # Fallback to AABB based OBB if robust OBB fails
        # print("Oriented BBox computation failed, falling back.")
        oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
    return axis_aligned_bbox, oriented_bbox


def color_by_instance_for_unik3d(pcds):
    """Assign unique colors to a list of PointClouds for UniK3D output."""
    if not pcds:
        return []
    cmap = matplotlib.colormaps.get_cmap( "turbo") # Or any other preferred colormap
    instance_colors = cmap(np.linspace(0, 1, len(pcds)))
    colored_pcds = []
    for i, pcd_original in enumerate(pcds):
        if pcd_original.has_points():
            pcd_copy = o3d.geometry.PointCloud(pcd_original) # Work on a copy
            pcd_copy.colors = o3d.utility.Vector3dVector(
                np.tile(instance_colors[i, :3], (len(pcd_copy.points), 1))
            )
            colored_pcds.append(pcd_copy)
        else:
            colored_pcds.append(o3d.geometry.PointCloud()) # Append empty if original was empty
    return colored_pcds


def oriented_bbox_to_center_euler_extent_for_unik3d(bbox_center, box_R, bbox_extent):
    """Convert OBB parameters to center, Euler angles, and extent for UniK3D output."""
    center = np.asarray(bbox_center)
    extent = np.asarray(bbox_extent)
    eulers = Rotation.from_matrix(box_R.copy()).as_euler("XYZ") # Default: extrinsic 'xyz'
    return center, eulers, extent


def axis_aligned_bbox_to_center_euler_extent_for_unik3d(min_coords, max_coords):
    """Convert AABB parameters to center, Euler angles, and extent for UniK3D output."""
    center = tuple((min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords))
    eulers = (0.0, 0.0, 0.0) # AABB is axis-aligned
    extent = tuple(abs(max_val - min_val) for min_val, max_val in zip(min_coords, max_coords))
    return center, eulers, extent

# --- End of Helper functions ---

warnings.filterwarnings("ignore") # Suppress warnings
def instantiate_model(model_name):
    type_ = model_name[0].lower()

    name = f"unik3d-vit{type_}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    # Set resolution level and interpolation mode as specified.
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model
# --- Helper functions ---
def prepare_llm_prompts_from_facts(facts, detection_list):
    batched_instructions = []
    for fact_instruction in facts:
        i_regions_found = re.findall(r"<region(\d+)>", fact_instruction)
        region_to_tag = {}
        valid_regions_in_fact = True
        for r_idx_str in i_regions_found:
            r_idx = int(r_idx_str)
            if 0 <= r_idx < len(detection_list):
                 region_to_tag[r_idx] = detection_list[r_idx]["class_name"]
            else:
                # Get a logger instance if available, otherwise print
                logger = setup_logger("prepare_llm_prompts") if OSDSYNTH_AVAILABLE else logging.getLogger("prepare_llm_prompts")
                logger.warning(f"Region index {r_idx} from fact '{fact_instruction}' is out of bounds for detection_list (len {len(detection_list)}).")
                valid_regions_in_fact = False
                break
        if not valid_regions_in_fact:
            continue

        object_references = []
        unique_region_indices = sorted(list(set(map(int, i_regions_found))))
        for r_idx in unique_region_indices:
            if r_idx in region_to_tag:
                 object_references.append(f"<region{r_idx}> {region_to_tag[r_idx]}")
        object_reference_str = ", ".join(object_references)
        new_instruction_for_llm = f"[Objects]: {object_reference_str}. [Description]: {fact_instruction}"
        batched_instructions.append(new_instruction_for_llm)
    return batched_instructions

def parse_qas_from_vqa_results(vqa_results):
    conversations = []
    for item in vqa_results:
        qa_pair = item[0]
        conversations.append(qa_pair)
    return conversations

LLM_HF_SYSTEM_PROMPT = r"""
You are a helpful assistant tasked with generating spatial reasoning-based questions and answers from provided descriptions of scenes.
Your response MUST be a single, valid JSON object. Do NOT include any text outside of this JSON object, such as "Here is the JSON:" or explanations.
The JSON object must have two keys: "Question" and "Answer".
The "Question" should be a string.
The "Answer" should be a string.

Always craft a question without directly revealing specific details from the description.
Always generate questions related to the description using <regionX>.
The description should always be used to answer and not leak into the question.
When mentioning the objects or regions, use <regionX> instead of the objects or regions.
Speak like you are the observer's perspective.
Always make sure all the description objects or regions are mentioned with <regionX> in the question.
Only mention each <regionX> once in the question.

Here are several examples of the input you will receive and the JSON output you MUST produce:

Input:
[Objects]: <region4> sofa, <region1> chair. [Description]: The path between the <region4> and <region1> is 1.5 meters.
Output:
{
    "Question": "You are a cleaning robot that is 1 meter wide. Now you are standing in a living room and see the image; you want to move from here to the door that leads to the backyard. Do you think I can go through the path between the <region4> and <region1>?",
    "Answer": "The path between <region4> and <region1> is 1.5 meters, so yes, the robot can go through the path between <region4> and <region1> since it is wider than the robot's width."
}

Input:
[Objects]: <region2> apple, <region3> orange. [Description]: <region2> is positioned on the left side of <region3>.
Output:
{
    "Question": "You see two fruits, an apple in <region2> and an orange in <region3>. Which one is more on the left side?",
    "Answer": "The apple in <region2> is more on the left."
}

Input:
[Objects]: <region0> book. [Description]: <region0> is 50 cm in width.
Output:
{
    "Question": "You are a librarian currently standing in front of a 40 cm width bookshelf, and you see <region0> that you want to place on the shelf. Can you determine if <region0> will fit on the shelf?",
    "Answer": "<region0> is 50 cm in width, so the shelf is not wide enough to hold a book of that size. Please find a larger shelf."
}

Now it's your turn!
"""


class GeneralizedSceneGraphGenerator:    

    def __init__(self, config_path="config/v2_hf_qwen.py", device="cuda",
                 llm_model_name_hf=None, llm_device_hf="auto"):
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.cfg = Config.fromfile(config_path)
        self.device = device
        self.logger = setup_logger(name="GeneralizedSceneGraphGeneratorHF")
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        if not OSDSYNTH_AVAILABLE:
            self.logger.error("osdsynth library components are not available. Cannot proceed.")
            raise ImportError("Failed to load osdsynth library.")

        self.segmenter = SegmentImage(self.cfg, self.logger, self.device)
        
        # Initialize with unik3d using the new function
        try:
            self.logger.info("Initializing UniK3D model...")
            self.unik3d_model = instantiate_model("Large")  # Example model size
            self.logger.info("Successfully initialized UniK3D model")
        except Exception as e:
            self.logger.error(f"Failed to initialize UniK3D model: {e}")
            raise RuntimeError("Could not initialize UniK3D model")

        self.captioner = CaptionImage(self.cfg, self.logger, self.device, init_lava=False)
        self.qa_prompter = QAPromptGenerator(self.cfg, self.logger, self.device)
        self.fact_prompter = FactPromptGenerator(self.cfg, self.logger, self.device)

        self.llm_pipeline = None
        self.llm_tokenizer_hf = None
        if llm_model_name_hf and HF_TRANSFORMERS_AVAILABLE:
            self.logger.info(f"Initializing Hugging Face LLM for: {llm_model_name_hf}")
            try:
                self.llm_tokenizer_hf = AutoTokenizer.from_pretrained(
                    llm_model_name_hf,
                    trust_remote_code=True
                )
                if self.llm_tokenizer_hf.pad_token is None:
                    if self.llm_tokenizer_hf.eos_token:
                        self.llm_tokenizer_hf.pad_token = self.llm_tokenizer_hf.eos_token
                        self.logger.info(f"Set tokenizer pad_token to eos_token ({self.llm_tokenizer_hf.eos_token})")
                    elif self.llm_tokenizer_hf.unk_token: # Fallback for padding
                        self.llm_tokenizer_hf.pad_token = self.llm_tokenizer_hf.unk_token
                        self.logger.warning(f"Set tokenizer pad_token to unk_token ({self.llm_tokenizer_hf.unk_token}). This might not be ideal.")
                    else:
                        self.logger.warning("Tokenizer has no pad_token, eos_token, or unk_token defined. Padding may be an issue or default to a model-specific behavior.")


                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=llm_model_name_hf,
                    tokenizer=self.llm_tokenizer_hf,
                    device_map=llm_device_hf,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                self.logger.info(f"Hugging Face LLM pipeline for {llm_model_name_hf} initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Hugging Face LLM pipeline: {e}", exc_info=True)
                self.llm_pipeline = None
                self.llm_tokenizer_hf = None
        elif llm_model_name_hf and not HF_TRANSFORMERS_AVAILABLE:
            self.logger.warning("llm_model_name_hf provided, but Hugging Face Transformers is not installed.")

        default_wis3d_folder = os.path.join(self.cfg.get("log_dir", "./temp_outputs/log"), f"Wis3D_Generalized_HF_{self.timestamp}")
        self.cfg.wis3d_folder = self.cfg.get("wis3d_folder", default_wis3d_folder)
        os.makedirs(self.cfg.wis3d_folder, exist_ok=True)
        self.cfg.vis = self.cfg.get("vis", False)

    unik3d_model = instantiate_model("Large") 
    def _override_config_and_reinit(self, **kwargs):
        reinit_segmenter = False
        reinit_reconstructor = False
        reinit_captioner = False
        for key, value in kwargs.items():
            parts = key.split('.')
            cfg_node = self.cfg
            changed = False
            try:
                for i, part in enumerate(parts[:-1]):
                    cfg_node = cfg_node[part]
                if cfg_node.get(parts[-1]) != value:
                    cfg_node[parts[-1]] = value
                    changed = True
            except KeyError:
                 cfg_node[parts[-1]] = value
                 changed = True
            if changed:
                if key.startswith("sam_") or key.startswith("box_") or key.startswith("text_") or \
                   key.startswith("nms_") or key == "class_set" or key == "remove_classes" or \
                   key.startswith("specified_tags") or key.startswith("grounding_dino_"):
                    reinit_segmenter = True
                if key.startswith("downsample_") or key.startswith("dbscan_") or \
                   key.startswith("perspective_model_") or key.startswith("min_points_"):
                    reinit_reconstructor = True
                if key.startswith("llava_") or key.startswith("global_qs_list"):
                    reinit_captioner = True
        if reinit_segmenter:
            self.segmenter = SegmentImage(self.cfg, self.logger, self.device)
        if reinit_reconstructor:
            init_models_flag = self.unik3d_model is not None
            self.unik3d_model = instantiate_model("Large")  # Example model size
        if reinit_captioner:
            init_lava_flag = self.captioner.llava_processor is not None
            self.captioner = CaptionImage(self.cfg, self.logger, self.device, init_lava=init_lava_flag)

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            image_bgr = image_input.copy()
        else:
            raise TypeError("image_input must be a file path (str) or a NumPy array (BGR).")
        h, w = image_bgr.shape[:2]
        if h == 0: raise ValueError("Image has zero height.")
        target_h = self.cfg.get("image_resize_height", 640) # Allow config override
        scale = target_h / h
        target_w = int(w * scale)
        image_bgr_resized = cv2.resize(image_bgr, (target_w, target_h))
        return image_bgr_resized

    def _get_object_classes(self, image_rgb_pil, custom_vocabulary=None):
        if custom_vocabulary:
            if not isinstance(custom_vocabulary, list) or not all(isinstance(s, str) for s in custom_vocabulary):
                raise ValueError("custom_vocabulary must be a list of strings.")
            if not custom_vocabulary:
                raise ValueError("custom_vocabulary list cannot be empty.")
            self.logger.info(f"Using custom vocabulary for segmentation: {custom_vocabulary}")
            return custom_vocabulary
        else:
            self.logger.info("Using Tag2Text for open-vocabulary class detection.")
            if self.segmenter.tagging_transform is None or self.segmenter.tagging_model is None:
                raise RuntimeError("Tagging model/transform not initialized in Segmenter.")
            img_tagging = image_rgb_pil.resize((384, 384))
            img_tagging_tensor = self.segmenter.tagging_transform(img_tagging).unsqueeze(0).to(self.device)
            classes = run_tagging_model(self.cfg, img_tagging_tensor, self.segmenter.tagging_model)
            if not classes:
                raise SkipImageException("Tag2Text detected no classes matching criteria.")
            self.logger.info(f"Tag2Text detected classes: {classes}")
            return classes

    def _segment_image(self, image_bgr, classes_to_detect):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb_pil = Image.fromarray(image_rgb)
        detections = self.segmenter.grounding_dino_model.predict_with_classes(
            image=image_bgr, classes=classes_to_detect,
            box_threshold=self.cfg.box_threshold, text_threshold=self.cfg.text_threshold,
        )
        if not hasattr(detections, 'class_id') or not detections.class_id.size > 0 or len(detections.xyxy) == 0:
            raise SkipImageException(f"No objects detected by GroundingDINO for classes: {classes_to_detect}")
        
        xyxy_tensor = torch.from_numpy(detections.xyxy).to(self.device if torch.cuda.is_available() else "cpu")
        confidence_tensor = torch.from_numpy(detections.confidence).to(self.device if torch.cuda.is_available() else "cpu")
        
        nms_idx = torchvision.ops.nms(xyxy_tensor, confidence_tensor, self.cfg.nms_threshold).cpu().numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx] # These are indices into classes_to_detect
        
        if len(detections.xyxy) == 0: raise SkipImageException("No detections remaining after NMS.")
        
        # Filter class_id that might be -1 (though predict_with_classes might prevent this)
        valid_idx = detections.class_id != -1 
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]
        
        if len(detections.xyxy) == 0: raise SkipImageException("No valid detections after NMS and class_id filtering.")

        detections.mask = get_sam_segmentation_from_xyxy(
            sam_predictor=self.segmenter.sam_predictor, image=image_rgb, xyxy=detections.xyxy
        )
        detections_dict = convert_detections_to_dict(detections, classes_to_detect) # Pass actual class names
        detections_dict = filter_detections(self.cfg, detections_dict, image_rgb)
        if len(detections_dict["xyxy"]) < 1: raise SkipImageException("No object detected after filtering.")
        detections_dict["subtracted_mask"], _ = mask_subtract_contained(
            detections_dict["xyxy"], detections_dict["mask"],
            th1=self.cfg.get("mask_contain_th1", 0.05),
            th2=self.cfg.get("mask_contain_th2", 0.05)
        )
        detections_dict = sort_detections_by_area(detections_dict)
        detections_dict = post_process_mask(detections_dict)
        detection_list = convert_detections_to_list(detections_dict, classes_to_detect) # Pass actual class names
        detection_list = crop_detections_with_xyxy(self.cfg, image_rgb_pil, detection_list)
        return detection_list

    def _process_common(self, image_input, custom_vocabulary=None, **kwargs):
        self._override_config_and_reinit(**kwargs)
        image_bgr = self._load_image(image_input)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image_rgb_pil = Image.fromarray(image_rgb) # This PIL image is used for _get_object_classes

        filename_prefix = "processed_image_" + self.timestamp
        if isinstance(image_input, str):
            filename_prefix = os.path.splitext(os.path.basename(image_input))[0] + "_" + self.timestamp

        # --- Segmentation ---
        # _get_object_classes expects PIL image
        object_classes = self._get_object_classes(Image.fromarray(image_rgb), custom_vocabulary)
        detection_list_initial = self._segment_image(image_bgr, object_classes) # Returns list of dicts
        if not detection_list_initial:
            raise SkipImageException("Segmentation resulted in no initial detections.")

        # --- UniK3D Point Cloud Generation ---
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad(): # Ensure no_grad for inference
            outputs = self.unik3d_model.infer(image_tensor, camera=None, normalize=True)
        # points_3d from UniK3D is typically (1, C, H, W) or (C, H, W) if C is spatial (x,y,z)
        # Assuming UniK3D output 'points' is (1, 3, H, W) where 3 is (x,y,z)
        points_3d_global = outputs["points"].squeeze().permute(1, 2, 0).cpu().numpy() # Shape: (H, W, 3)

        # --- Wis3D Visualization Setup (if enabled) ---
        wis3d_instance = None
        if self.cfg.get("vis", False):
            wis3d_instance = Wis3D(self.cfg.wis3d_folder, filename_prefix)
            # Add global point cloud from UniK3D
            # Ensure points_3d_global and image_rgb have compatible shapes for coloring
            if points_3d_global.shape[:2] == image_rgb.shape[:2]:
                wis3d_instance.add_point_cloud(
                    vertices=points_3d_global.reshape((-1, 3)),
                    colors=image_rgb.reshape(-1, 3), # Use original image colors
                    name="unik3d_global_scene_pts"
                )
            else:
                self.logger.warning("Global point cloud and image dimensions mismatch for Wis3D coloring.")

        # --- Process each detection for 3D information ---
        valid_detections = []
        min_initial_pts = self.cfg.get("min_points_threshold", 20)
        min_processed_pts = self.cfg.get("min_points_threshold_after_denoise", 10)
        min_bbox_volume = self.cfg.get("bbox_min_volume_threshold", 1e-6)

        for det_idx, det in enumerate(detection_list_initial):
            # 'subtracted_mask' is expected to be (H, W) boolean from _segment_image
            object_mask = det["subtracted_mask"] 
            
            if not np.any(object_mask): # Skip if mask is empty
                self.logger.debug(f"Skipping det {det_idx} ({det.get('class_name', 'N/A')}) due to empty mask.")
                continue

            # Extract points and colors for the current object using its mask
            obj_points_from_global = points_3d_global[object_mask]
            obj_colors_from_global = image_rgb[object_mask] / 255.0 # Normalize colors to [0,1]

            if len(obj_points_from_global) < min_initial_pts:
                self.logger.debug(f"Skipping det {det_idx} ({det.get('class_name', 'N/A')}): "
                                f"Too few initial points ({len(obj_points_from_global)} < {min_initial_pts}).")
                continue
                
            # Perturb points slightly to avoid co-linearity issues for some o3d operations
            obj_points_from_global += np.random.normal(0, self.cfg.get("pcd_perturb_std", 1e-3), obj_points_from_global.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points_from_global)
            pcd.colors = o3d.utility.Vector3dVector(obj_colors_from_global)
            
            # Process the object's point cloud (denoise, downsample)
            processed_pcd = process_pcd_for_unik3d(self.cfg, pcd)

            if not processed_pcd.has_points() or len(processed_pcd.points) < min_processed_pts:
                self.logger.debug(f"Skipping det {det_idx} ({det.get('class_name', 'N/A')}): "
                                f"Too few points after processing ({len(processed_pcd.points)} < {min_processed_pts}).")
                continue

            # Get bounding boxes
            axis_aligned_bbox, oriented_bbox = get_bounding_box_for_unik3d(self.cfg, processed_pcd)

            if axis_aligned_bbox.is_empty() or axis_aligned_bbox.volume() < min_bbox_volume:
                self.logger.debug(f"Skipping det {det_idx} ({det.get('class_name', 'N/A')}): "
                                f"BBox volume too small ({axis_aligned_bbox.volume()}).")
                continue
                
            det["pcd"] = processed_pcd # Store the processed PCD
            det["axis_aligned_bbox"] = axis_aligned_bbox
            det["oriented_bbox"] = oriented_bbox
            # det["points3d"] = np.asarray(processed_pcd.points) # If raw numpy points are still needed elsewhere
            # det["image_crop"] is from _segment_image, could be useful
            valid_detections.append(det)

        if not valid_detections:
            raise SkipImageException("No valid 3D objects found after processing.")

        # --- Wis3D Visualization for valid objects (if enabled) ---
        if wis3d_instance and valid_detections:
            object_pcds_for_vis = [d["pcd"] for d in valid_detections]
            instance_colored_pcds = color_by_instance_for_unik3d(object_pcds_for_vis)

            for i, det_data in enumerate(valid_detections):
                obj_pcd_colored = instance_colored_pcds[i]
                class_name = det_data.get("class_name", f"object_{i}")
                
                if obj_pcd_colored.has_points():
                    wis3d_instance.add_point_cloud(
                        vertices=np.asarray(obj_pcd_colored.points),
                        colors=np.asarray(obj_pcd_colored.colors),
                        name=f"{i:02d}_{class_name}_unik3d_pts"
                    )

                # Add BBoxes to Wis3D
                aa_bbox = det_data["axis_aligned_bbox"]
                if not aa_bbox.is_empty():
                    aa_center, aa_eulers, aa_extent = axis_aligned_bbox_to_center_euler_extent_for_unik3d(
                        aa_bbox.get_min_bound(), aa_bbox.get_max_bound()
                    )
                    wis3d_instance.add_boxes(
                        positions=np.array([aa_center]), 
                        eulers=np.array([aa_eulers]), 
                        extents=np.array([aa_extent]), 
                        name=f"{i:02d}_{class_name}_unik3d_aa_bbox"
                    )
                
                or_bbox = det_data["oriented_bbox"]
                # Check if oriented_bbox is valid (e.g. extent is not zero)
                if not or_bbox.is_empty() and np.all(np.array(or_bbox.extent) > 1e-6):
                    or_center, or_eulers, or_extent = oriented_bbox_to_center_euler_extent_for_unik3d(
                        or_bbox.center, or_bbox.R, or_bbox.extent
                    )
                    wis3d_instance.add_boxes(
                        positions=np.array([or_center]), 
                        eulers=np.array([or_eulers]), 
                        extents=np.array([or_extent]), 
                        name=f"{i:02d}_{class_name}_unik3d_or_bbox"
                    )

        # --- Captioning (on valid detections that have 3D info) ---
        captioned_detections = self.captioner.process_local_caption(valid_detections)
        
        return captioned_detections, filename_prefix

    def generate_facts(self, image_input, custom_vocabulary=None, run_llm_rephrase=False, **kwargs):
        try:
            detection_list, filename_prefix = self._process_common(image_input, custom_vocabulary, **kwargs)
            if not detection_list:
                self.logger.warning("Common processing failed to produce detections for fact generation.")
                return None, [], []

            template_facts = self.fact_prompter.evaluate_predicates_on_pairs(detection_list)
            rephrased_qas = []
            if run_llm_rephrase and template_facts:
                if not self.llm_pipeline:
                    self.logger.warning("LLM pipeline not initialized. Skipping LLM rephrasing.")
                else:
                    llm_prompts = prepare_llm_prompts_from_facts(template_facts, detection_list)
                    if llm_prompts:
                        rephrased_qas = self._run_llm_rephrasing_hf(llm_prompts)
            
            if self.cfg.get("vis", False): # Check cfg for vis, not just kwargs
                 self.logger.info(f"Wis3D visualization potentially saved for {filename_prefix} in {self.cfg.wis3d_folder}")
            return detection_list, template_facts, rephrased_qas
        except SkipImageException as e:
            self.logger.warning(f"Fact generation skipped for image: {e}")
            return None, [], []
        except Exception as e:
            self.logger.error(f"Error during fact generation: {e}", exc_info=True)
            return None, [], []

    def generate_qa(self, image_input, custom_vocabulary=None, **kwargs):
        try:
            detection_list, filename_prefix = self._process_common(image_input, custom_vocabulary, **kwargs)
            if not detection_list:
                self.logger.warning("Common processing failed to produce detections for QA generation.")
                return None, []
            
            vqa_results = self.qa_prompter.evaluate_predicates_on_pairs(detection_list)
            template_qas = parse_qas_from_vqa_results(vqa_results)
            
            if self.cfg.get("vis", False):
                 self.logger.info(f"Wis3D visualization potentially saved for {filename_prefix} in {self.cfg.wis3d_folder}")
            return detection_list, template_qas
        except SkipImageException as e:
            self.logger.warning(f"QA generation skipped for image: {e}")
            return None, []
        except Exception as e:
            self.logger.error(f"Error during QA generation: {e}", exc_info=True)
            return None, []

    def _parse_llm_json_output(self, llm_output_text):
        # Strip leading/trailing whitespace before any parsing
        llm_output_text_stripped = llm_output_text.strip()

        # Try to find JSON block, including ```json ... ```
        match_json_block = re.search(r"```json\s*(\{.*?\})\s*```", llm_output_text_stripped, re.DOTALL)
        if match_json_block:
            json_str = match_json_block.group(1)
        else:
            # If no markdown block, try to find first '{' and last '}'
            first_brace = llm_output_text_stripped.find('{')
            last_brace = llm_output_text_stripped.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
                json_str = llm_output_text_stripped[first_brace : last_brace+1]
            else: # Fallback: if it starts with { and ends with }, assume it's all JSON
                if llm_output_text_stripped.startswith("{") and llm_output_text_stripped.endswith("}"):
                    json_str = llm_output_text_stripped
                else:
                    self.logger.warning(f"Could not clearly identify JSON block in LLM output: {llm_output_text_stripped[:200]}")
                    return None
        try:
            # Clean internal newlines that are not part of string literals
            # This is a heuristic. A more robust parser might be needed for complex cases.
            # json_str_cleaned = re.sub(r'(?<!\\)\n', ' ', json_str) # Careful with this, might break valid newlines in strings
            json_str_cleaned = json_str # For now, assume newlines inside JSON string are intended or handled by json.loads
            parsed_json = json.loads(json_str_cleaned)
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSONDecodeError for string: '{json_str[:200]}...'. Error: {e}. Original text: {llm_output_text_stripped[:200]}")
            return None

    def _run_llm_rephrasing_hf(self, llm_prompts):
        if not self.llm_pipeline or not self.llm_tokenizer_hf:
            self.logger.warning("Hugging Face LLM pipeline or tokenizer not available. Skipping rephrasing.")
            return []
        
        rephrased_conversations = []
        for user_prompt_text in llm_prompts:
            messages = [
                {"role": "system", "content": LLM_HF_SYSTEM_PROMPT},
                {"role": "user", "content": f"Input:\n{user_prompt_text}\nOutput:"}
            ]

            try:
                prompt_for_llm = self.llm_tokenizer_hf.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.logger.info(f"DEBUG: Explicitly formatted prompt for LLM:\n-----\n{prompt_for_llm}\n-----")
            except Exception as e_template:
                self.logger.error(f"Failed to apply chat template: {e_template}. Skipping this prompt.", exc_info=True)
                continue

            max_retries = self.cfg.get("llm_max_retries", 3)
            success = False
            q_final, a_final = None, None

            for attempt in range(max_retries):
                try:
                    terminators = []
                    # Qwen/Llama specific EOS: <|im_end|> or <|eot_id|>
                    # It's safer to rely on tokenizer.eos_token_id if it's correctly set by the tokenizer for the model
                    # or add known terminators manually.
                    if self.llm_tokenizer_hf.eos_token_id is not None:
                        terminators.append(self.llm_tokenizer_hf.eos_token_id)
                    
                    # For Qwen, <|im_end|> is critical
                    im_end_token_id = self.llm_tokenizer_hf.convert_tokens_to_ids("<|im_end|>")
                    if isinstance(im_end_token_id, int) and im_end_token_id not in terminators:
                        terminators.append(im_end_token_id)
                    # For Llama3, <|eot_id|> is also important
                    eot_id_llama = self.llm_tokenizer_hf.convert_tokens_to_ids("<|eot_id|>")
                    if isinstance(eot_id_llama, int) and eot_id_llama not in terminators:
                         terminators.append(eot_id_llama)


                    if not terminators: # Fallback
                        self.logger.warning("No specific EOS token IDs found. Relying on default pipeline behavior.")
                        eos_pipeline_arg = None # Let pipeline decide or use its configured default
                    else:
                        eos_pipeline_arg = terminators
                        self.logger.info(f"DEBUG: Using EOS token IDs for generation: {eos_pipeline_arg}")

                    pipeline_args = {
                        "max_new_tokens": self.cfg.get("llm_max_new_tokens", 512),
                        "temperature": self.cfg.get("llm_temperature", 0.1),
                        "do_sample": True,
                    }
                    if eos_pipeline_arg is not None: # Only add if we have valid terminators
                        pipeline_args["eos_token_id"] = eos_pipeline_arg
                    
                    if self.llm_tokenizer_hf.pad_token_id is not None:
                        pipeline_args["pad_token_id"] = self.llm_tokenizer_hf.pad_token_id
                    else: # If pad_token_id is None, but eos_token_id is available, use it for padding
                        if self.llm_tokenizer_hf.eos_token_id is not None:
                            pipeline_args["pad_token_id"] = self.llm_tokenizer_hf.eos_token_id
                            self.logger.info(f"Using tokenizer.eos_token_id ({self.llm_tokenizer_hf.eos_token_id}) as pad_token_id.")
                        else:
                            self.logger.warning("pad_token_id and eos_token_id are None in tokenizer. Pipeline might default or error if padding is implicitly needed.")


                    self.logger.info(f"DEBUG Attempt {attempt + 1}: Calling LLM pipeline with args: {pipeline_args}")
                    generated_outputs = self.llm_pipeline(prompt_for_llm, **pipeline_args)
                    
                    self.logger.info(f"DEBUG Attempt {attempt + 1}: RAW LLM Output type: {type(generated_outputs)}")
                    self.logger.info(f"DEBUG Attempt {attempt + 1}: RAW LLM Output value: {str(generated_outputs)[:1000]}")

                    actual_llm_generation = None
                    if generated_outputs and isinstance(generated_outputs, list) and generated_outputs[0]:
                        if "generated_text" in generated_outputs[0]:
                            full_response_with_prompt = generated_outputs[0]["generated_text"]
                            if isinstance(full_response_with_prompt, str):
                                if full_response_with_prompt.startswith(prompt_for_llm):
                                    actual_llm_generation = full_response_with_prompt[len(prompt_for_llm):]
                                    self.logger.info(f"DEBUG Attempt {attempt + 1}: Stripped input. Actual LLM generation: {str(actual_llm_generation)[:500]}")
                                else:
                                    actual_llm_generation = full_response_with_prompt
                                    self.logger.info(f"DEBUG Attempt {attempt + 1}: LLM output did not start with input prompt. Using full output as generation: {str(actual_llm_generation)[:500]}")
                            else:
                                self.logger.warning(f"DEBUG Attempt {attempt+1}: 'generated_text' content is not a string. Type: {type(full_response_with_prompt)}")
                        else:
                            self.logger.warning(f"DEBUG Attempt {attempt+1}: 'generated_text' key not found in pipeline output.")
                    else:
                        self.logger.warning(f"DEBUG Attempt {attempt+1}: Unexpected pipeline output format or empty output.")

                    if actual_llm_generation is not None and isinstance(actual_llm_generation, str):
                        # Clean known trailing EOS tokens if they are not part of the JSON
                        for known_eos in ["<|im_end|>", "<|eot_id|>", self.llm_tokenizer_hf.eos_token]:
                            if known_eos and actual_llm_generation.strip().endswith(known_eos):
                                actual_llm_generation = actual_llm_generation.strip()[:-len(known_eos)].strip()
                                self.logger.info(f"DEBUG: Stripped trailing '{known_eos}' from LLM generation.")
                        
                        self.logger.info(f"DEBUG Attempt {attempt+1}: Final actual_llm_generation for parsing: {actual_llm_generation[:500]}")
                        json_response = self._parse_llm_json_output(actual_llm_generation) # Already stripped in _parse

                        if not json_response:
                            self.logger.warning(f"LLM attempt {attempt + 1}: Could not parse JSON from response: {actual_llm_generation[:200]}...")
                            continue
                    else:
                        self.logger.warning(f"LLM attempt {attempt + 1}: actual_llm_generation is None or not a string. Original output: {str(generated_outputs)[:200]}")
                        continue
                    
                    question, answer = json_response.get("Question"), json_response.get("Answer")
                    if question is None or answer is None:
                        self.logger.warning(f"LLM response missing Question or Answer. Parsed JSON: {json_response}")
                        continue

                    question = question[2:] if question.startswith(". ") else question
                    answer = answer[2:] if answer.startswith(". ") else answer
                    prompt_tags = set(re.findall(r"<region\d+>", user_prompt_text))
                    question_tags = set(re.findall(r"<region\d+>", question))
                    
                    print('-----Count-----')
                    print(len(question))
                    print(len(answer))
                    print(len(prompt_tags))
                    print(len(question_tags))
                    print(prompt_tags)
                    print(question_tags)

                    if prompt_tags.issubset(question_tags) and question_tags.issubset(prompt_tags):
                        if all(question.count(tag) == 1 for tag in prompt_tags):
                            q_final, a_final = question, answer
                            success = True
                            break
                        else:
                            self.logger.debug(f"LLM attempt {attempt + 1}: <regionX> appeared >1 times in question for prompt '{user_prompt_text[:50]}...'")
                    else:
                        self.logger.debug(f"LLM attempt {attempt + 1}: <regionX> mismatch for prompt '{user_prompt_text[:50]}...'. Prompt tags: {prompt_tags}, Q tags: {question_tags}")
                
                except Exception as e:
                    self.logger.warning(f"LLM rephrase attempt {attempt + 1} for prompt '{user_prompt_text[:50]}...' failed: {e}", exc_info=True)
            
            if success:
                rephrased_conversations.append((q_final, a_final))
                self.logger.info(f"LLM Rephrased => Q: {q_final} || A: {a_final}")
            else:
                 self.logger.warning(f"LLM failed for prompt: '{user_prompt_text[:100]}...' after {max_retries} attempts.")
        return rephrased_conversations

    def __del__(self):
        if hasattr(self, 'llm_pipeline') and self.llm_pipeline is not None:
            if hasattr(self.llm_pipeline, 'model') and self.llm_pipeline.model is not None:
                if hasattr(self.llm_pipeline.model, 'cpu'):
                    try:
                        self.llm_pipeline.model.cpu()
                        self.logger.info("Moved LLM model to CPU.")
                    except Exception as e:
                        self.logger.warning(f"Could not move LLM model to CPU: {e}")
                del self.llm_pipeline.model
            del self.llm_pipeline
            self.llm_pipeline = None
            self.llm_tokenizer_hf = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Attempted to release LLM resources.")

# --- Example Usage (Update config for Qwen2 or Llama3) ---
if __name__ == "__main__":
    # 1. DEFINE YOUR CONFIG FILE PATH
    #    Ensure this file exists and is COMPLETE.
    #    It should contain all vision parameters copied from a working config,
    #    plus the LLM parameters updated for your chosen model.
    CONFIG_FILE_NAME = "v2_hf_llm.py" # Generic name, ensure content matches chosen LLM
    config_dir = "configs"
    config_file_path = os.path.join(config_dir, CONFIG_FILE_NAME)

    # 2. CREATE/UPDATE THE CONFIG FILE (if it doesn't exist or needs updating)
    #    This is a TEMPLATE. You MUST ensure it's complete.
    #    Replace "Qwen/Qwen2-7B-Instruct" with your actual model if different.
    #    Adjust llm_max_new_tokens based on model and expected output length.

    # 3. CREATE A DUMMY DEMO IMAGE (if it doesn't exist)
    demo_image_dir = "./demo_images"
    demo_image_path = os.path.join(demo_image_dir, "indoor.png")
    if not os.path.exists(demo_image_path):
        os.makedirs(demo_image_dir, exist_ok=True)
        dummy_img_main = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img_main, (100,100), (200,200), (0,255,0), -1) # Green box
        cv2.rectangle(dummy_img_main, (300,200), (450,350), (0,0,255), -1) # Red box
        cv2.putText(dummy_img_main, "Object1", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(dummy_img_main, "Object2", (300, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(demo_image_path, dummy_img_main)
        print(f"Created/verified dummy image at {demo_image_path}")

    # 4. INITIALIZE AND RUN
    print(f"Initializing GeneralizedSceneGraphGenerator with config: {config_file_path}")
    try:
        temp_cfg_main = Config.fromfile(config_file_path)
        llm_name_for_init_main = temp_cfg_main.get("llm_model_name_hf")
        if not llm_name_for_init_main:
            raise ValueError("llm_model_name_hf not found in config file.")
        
        print(f"Attempting to use LLM: {llm_name_for_init_main}")

        generator_main = GeneralizedSceneGraphGenerator(
            config_path=config_file_path,
            llm_model_name_hf=llm_name_for_init_main,
            llm_device_hf="auto" # "cuda" for single GPU, "auto" for device_map
        )
    except Exception as e_init:
        print(f"FATAL ERROR during generator initialization: {e_init}")
        import traceback
        traceback.print_exc()
        exit()
    
    # Use a real image path for actual testing if desired
    # my_actual_image_path = "./path/to/your/image.jpg"
    # current_image_to_process = my_actual_image_path if os.path.exists(my_actual_image_path) else demo_image_path
    current_image_to_process = demo_image_path


    print(f"\n--- Generating Facts for {current_image_to_process} with LLM ---")
    try:
        run_llm_rephrase = True
        # Example: Use Tag2Text by setting custom_vocabulary=None
        detections_f, facts, rephrased_qas_f = generator_main.generate_facts(
            current_image_to_process,
            custom_vocabulary=None, 
            run_llm_rephrase=True,
            vis=True # Set to True for Wis3D output
        )

        if detections_f:
            print(f"Generated {len(detections_f)} detections.")
            print(f"Generated {len(facts)} template facts:")
            for i, fact_str in enumerate(facts):
                print(f"  Fact {i+1}: {fact_str}")
            if rephrased_qas_f:
                print(f"Generated {len(rephrased_qas_f)} LLM-rephrased QAs from facts:")
                for i_qa, (q, a) in enumerate(rephrased_qas_f):
                    print(f"  LLM QA {i_qa+1}: Q: {q} || A: {a}")
            elif run_llm_rephrase:
                print("LLM rephrasing was enabled, but no QAs were generated.")
        else:
            print("No facts generated or detections found.")
            
    except SkipImageException as e_skip:
        print(f"Skipped image processing: {e_skip}")
    except Exception as e_facts:
        print(f"Error in fact generation example (LLM): {e_facts}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    del generator_main
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nProcessing complete.")


