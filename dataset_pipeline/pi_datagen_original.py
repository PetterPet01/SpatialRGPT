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
# import os # Already imported
import shutil
# import time # Already imported
from datetime import datetime
from math import pi, tan, radians  # Added tan, radians for focal length calculation

import gradio as gr
# import numpy as np # Already imported
# import torch # Already imported
import trimesh
# from PIL import Image # Already imported

# unik3d.models import UniK3D # REMOVED
# unik3d.utils.camera import OPENCV, Fisheye624, Pinhole, Spherical # REMOVED
import open3d as o3d

# import open3d as o3d # Should already be there
from wis3d import Wis3D
import matplotlib  # For color_by_instance
from scipy.spatial.transform import Rotation  # For oriented_bbox_to_center_euler_extent
from collections import Counter  # For pcd_denoise_dbscan

# YOLOE CHANGE: Import YOLOE
from ultralytics import YOLOE

# DEPTH ANYTHING V2 CHANGE: Import DepthAnythingV2
try:
    from depth_anything_v2.dpt import DepthAnythingV2

    DEPTH_ANYTHING_V2_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"Failed to import DepthAnythingV2: {e}. "
        "Ensure depth_anything_v2 directory is in PYTHONPATH and dependencies are installed. "
        "Depth estimation functionalities will be unavailable.")
    DEPTH_ANYTHING_V2_AVAILABLE = False


    class DepthAnythingV2:  # Dummy class if not available
        def __init__(self, *args, **kwargs): pass

        def load_state_dict(self, *args, **kwargs): pass

        def to(self, *args, **kwargs): return self

        def eval(self): return self

        def infer_image(self, *args, **kwargs): raise NotImplementedError("DepthAnythingV2 not available")

# OSDSUTILS imports (ensure osdsynth is in PYTHONPATH or installed)
try:
    from osdsynth.processor.captions import CaptionImage
    from osdsynth.processor.pointcloud import \
        PointCloudReconstruction  # Though not directly used here, keep for osdsynth completeness
    from osdsynth.processor.prompt import PromptGenerator as QAPromptGenerator
    from osdsynth.processor.instruction import PromptGenerator as FactPromptGenerator
    from osdsynth.utils.logger import SkipImageException, setup_logger

    OSDSYNTH_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"Failed to import osdsynth components: {e}. Ensure osdsynth is correctly installed and in PYTHONPATH.")
    OSDSYNTH_AVAILABLE = False


    # Add dummy classes if osdsynth is not available to allow script to be parsed
    class CaptionImage:
        pass


    class PointCloudReconstruction:
        pass


    class QAPromptGenerator:
        pass


    class FactPromptGenerator:
        pass


    class SkipImageException(Exception):
        pass


    def setup_logger(name="dummy"):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:  # Avoid duplicate handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


    def crop_detections_with_xyxy(cfg, image_pil, detections_list):
        # Dummy version if osdsynth not available
        warnings.warn("Using dummy crop_detections_with_xyxy as osdsynth is not available.")
        for det in detections_list:
            det["image_crop"] = None  # Or some placeholder PIL image
        return detections_list

# Hugging Face Transformers imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    warnings.warn("Hugging Face Transformers not found. LLM rephrasing will not be available.")


def crop_detections_with_xyxy(cfg, image,
                              detections_list):  # This is the local one, preferred if OSDSynth one isn't loaded
    for idx, detection in enumerate(detections_list):
        x1, y1, x2, y2 = detection["xyxy"]
        image_crop, mask_crop = crop_image_and_mask(image, detection["mask"], x1, y1, x2, y2, padding=10)
        # Assuming `masking_option` logic for `image_crop_modified` is handled if needed,
        # or that `image_crop` is sufficient. The original did not use `image_crop_modified` directly
        # in the main flow after this function, but kept for compatibility if osdsynth version was more complex.
        # For simplicity, just assign image_crop and mask_crop if the more complex modification is not required here.
        detections_list[idx]["image_crop"] = image_crop
        detections_list[idx]["mask_crop"] = mask_crop
        # If blackout/red_outline is needed:
        # if cfg.masking_option == "blackout":
        #     image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop) # Define these helpers if used
        # elif cfg.masking_option == "red_outline":
        #     image_crop_modified = draw_red_outline(image_crop, mask_crop) # Define these helpers if used
        # else:
        #     image_crop_modified = image_crop
        # detections_list[idx]["image_crop_modified"] = image_crop_modified
    return detections_list


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """Crop the image and mask with some padding.
    """
    image_np = np.array(image)
    if image_np.shape[:2] != mask.shape:
        print(f"Initial shape mismatch: Image shape {image_np.shape} != Mask shape {mask.shape}")
        # Attempt to resize mask if it's a common issue like 1-pixel difference due to rounding
        if abs(image_np.shape[0] - mask.shape[0]) <= 2 and abs(image_np.shape[1] - mask.shape[1]) <= 2:
            print(f"Attempting to resize mask from {mask.shape} to {image_np.shape[:2]}")
            mask = cv2.resize(mask.astype(np.uint8), (image_np.shape[1], image_np.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            return None, None

    x1_padded = max(0, int(round(x1)) - padding)
    y1_padded = max(0, int(round(y1)) - padding)
    x2_padded = min(image_np.shape[1], int(round(x2)) + padding)
    y2_padded = min(image_np.shape[0], int(round(y2)) + padding)

    image_crop_np = image_np[y1_padded:y2_padded, x1_padded:x2_padded]
    mask_crop_np = mask[y1_padded:y2_padded, x1_padded:x2_padded]

    if image_crop_np.shape[:2] != mask_crop_np.shape:
        print(
            "Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(
                image_crop_np.shape, mask_crop_np.shape
            )
        )
        return None, None

    image_crop_pil = Image.fromarray(image_crop_np)
    return image_crop_pil, mask_crop_np


# --- Definition of the DetectedObject class --- (NO CHANGE)
class DetectedObject:
    def __init__(self,
                 class_name: str,
                 description: str,
                 segmentation_mask_2d: np.ndarray,
                 bounding_box_2d: np.ndarray,
                 point_cloud_3d: o3d.geometry.PointCloud,
                 bounding_box_3d_oriented: o3d.geometry.OrientedBoundingBox,
                 bounding_box_3d_axis_aligned: o3d.geometry.AxisAlignedBoundingBox,
                 image_crop_pil: Image.Image = None
                 ):
        self.class_name = class_name
        self.description = description
        self.segmentation_mask_2d = segmentation_mask_2d
        self.bounding_box_2d = bounding_box_2d
        self.point_cloud_3d = point_cloud_3d
        self.bounding_box_3d_oriented = bounding_box_3d_oriented
        self.bounding_box_3d_axis_aligned = bounding_box_3d_axis_aligned
        self.image_crop_pil = image_crop_pil

    def __repr__(self):
        num_points = len(self.point_cloud_3d.points) if self.point_cloud_3d and self.point_cloud_3d.has_points() else 0
        return (f"<DetectedObject: {self.class_name} "
                f"(Desc: '{self.description[:30]}...'), "
                f"2D_bbox: {self.bounding_box_2d.tolist()}, "
                f"Mask_Shape: {self.segmentation_mask_2d.shape if self.segmentation_mask_2d is not None else 'N/A'}, "
                f"3D_pts: {num_points}, "
                f"3D_OBB_center: {self.bounding_box_3d_oriented.center.tolist() if self.bounding_box_3d_oriented else 'N/A'}>")


# --- Helper functions for Point Cloud Processing and Visualization ---
# Renamed for clarity, but logic is general
def process_pcd_generic(cfg, pcd, run_dbscan=True):
    if not pcd.has_points() or len(pcd.points) == 0: return pcd
    try:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=cfg.get("pcd_sor_neighbors", 20),
                                                std_ratio=cfg.get("pcd_sor_std_ratio", 1.5))
    except RuntimeError as e:
        pass  # Can happen with too few points
    if not pcd.has_points() or len(pcd.points) == 0: return pcd
    voxel_size = cfg.get("pcd_voxel_size", 0.01)
    if voxel_size > 0: pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if not pcd.has_points() or len(pcd.points) == 0: return pcd
    if cfg.get("dbscan_remove_noise", True) and run_dbscan:
        pcd = pcd_denoise_dbscan_generic(pcd, eps=cfg.get("dbscan_eps", 0.05),
                                         min_points=cfg.get("dbscan_min_points", 10))
    return pcd


def pcd_denoise_dbscan_generic(pcd: o3d.geometry.PointCloud, eps=0.05, min_points=10) -> o3d.geometry.PointCloud:
    if not pcd.has_points() or len(pcd.points) < min_points: return pcd
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except RuntimeError as e:
        return pcd  # Can happen with certain point configurations
    counts = Counter(labels);
    if -1 in counts: del counts[-1]  # Remove noise label
    if not counts: return o3d.geometry.PointCloud()  # No clusters found
    largest_cluster_label = counts.most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    if len(largest_cluster_indices) < min_points: return o3d.geometry.PointCloud()  # Largest cluster too small
    return pcd.select_by_index(largest_cluster_indices)


def get_bounding_box_generic(cfg, pcd):
    if not pcd.has_points() or len(pcd.points) < 3:  # Need at least 3 points for OBB
        aabb = o3d.geometry.AxisAlignedBoundingBox();
        obb = o3d.geometry.OrientedBoundingBox()
        return aabb, obb
    axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()
    try:
        oriented_bbox = pcd.get_oriented_bounding_box(robust=cfg.get("obb_robust", True))
    except RuntimeError:  # Fallback if robust OBB fails
        oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
    return axis_aligned_bbox, oriented_bbox


def color_by_instance_generic(pcds):
    if not pcds: return []
    cmap = matplotlib.colormaps.get_cmap("turbo")  # A perceptually uniform colormap
    instance_colors = cmap(np.linspace(0, 1, len(pcds)))
    colored_pcds = []
    for i, pcd_original in enumerate(pcds):
        if pcd_original.has_points():
            pcd_copy = o3d.geometry.PointCloud(pcd_original)  # Work on a copy
            pcd_copy.colors = o3d.utility.Vector3dVector(np.tile(instance_colors[i, :3], (len(pcd_copy.points), 1)))
            colored_pcds.append(pcd_copy)
        else:
            colored_pcds.append(o3d.geometry.PointCloud())  # Append empty PCD if original was empty
    return colored_pcds


def oriented_bbox_to_center_euler_extent_generic(bbox_center, box_R, bbox_extent):
    center = np.asarray(bbox_center);
    extent = np.asarray(bbox_extent)
    eulers = Rotation.from_matrix(box_R.copy()).as_euler("XYZ")  # Ensure copy for safety
    return center, eulers, extent


def axis_aligned_bbox_to_center_euler_extent_generic(min_coords, max_coords):
    center = tuple((min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords))
    eulers = (0.0, 0.0, 0.0)  # Axis-aligned means no rotation relative to world axes
    extent = tuple(abs(max_val - min_val) for min_val, max_val in zip(min_coords, max_coords))
    return center, eulers, extent


# --- End of Helper functions ---

warnings.filterwarnings("ignore")


# instantiate_model for UniK3D (REMOVED)
# def instantiate_model(model_name):
#     ...

# prepare_llm_prompts_from_facts (NO CHANGE)
def prepare_llm_prompts_from_facts(facts, detection_list_dicts):
    batched_instructions = []
    for fact_instruction in facts:
        i_regions_found = re.findall(r"<region(\d+)>", fact_instruction)
        region_to_tag = {}
        valid_regions_in_fact = True
        for r_idx_str in i_regions_found:
            r_idx = int(r_idx_str)
            if 0 <= r_idx < len(detection_list_dicts):
                region_to_tag[r_idx] = detection_list_dicts[r_idx]["class_name"]
            else:
                current_logger_module = setup_logger(
                    "prepare_llm_prompts") if OSDSYNTH_AVAILABLE else logging.getLogger("prepare_llm_prompts")
                current_logger_module.warning(
                    f"Region index {r_idx} from fact '{fact_instruction}' is out of bounds for detection_list (len {len(detection_list_dicts)}).")
                valid_regions_in_fact = False
                break
        if not valid_regions_in_fact: continue
        object_references = []
        unique_region_indices = sorted(list(set(map(int, i_regions_found))))
        for r_idx in unique_region_indices:
            if r_idx in region_to_tag: object_references.append(f"<region{r_idx}> {region_to_tag[r_idx]}")
        object_reference_str = ", ".join(object_references)
        new_instruction_for_llm = f"[Objects]: {object_reference_str}. [Description]: {fact_instruction}"
        batched_instructions.append(new_instruction_for_llm)
    return batched_instructions


# parse_qas_from_vqa_results (NO CHANGE)
def parse_qas_from_vqa_results(vqa_results):
    conversations = []
    for item in vqa_results:
        qa_pair = item[0]  # Assuming VQA result item is a list/tuple where the first element is the QA pair
        conversations.append(qa_pair)
    return conversations


LLM_HF_SYSTEM_PROMPT = r"""You are an expert visual language model. Your task is to rephrase a [Description] about [Objects] into a natural question and answer pair.
The [Objects] section provides a list of objects referenced in the [Description], along with their unique identifiers like <region0>, <region1>, etc.
The [Description] is a factual statement about these objects and their relationships.
Your goal is to create a question that:
1.  Is natural and conversational.
2.  Directly asks about the information presented in the [Description].
3.  Critically, **MUST** include **ALL** the <regionN> identifiers mentioned in the original [Objects] list, and **EACH** <regionN> identifier must appear **EXACTLY ONCE** in the question. Do not introduce new <regionN> identifiers.
The answer should:
1.  Directly and truthfully answer the question based **ONLY** on the [Description].
2.  Be concise and to the point.
3.  **NOT** include any <regionN> identifiers.
Output the question and answer in a JSON format, like this:
```json
{
"Question": "Your rephrased question with <regionN> tags here.",
"Answer": "Your concise answer here."
}
```
Example:
Input:
[Objects]: <region0> person, <region1> dog. [Description]: <region0> is walking the <region1>.
Output:
```json
{
"Question": "What is the person <region0> doing with the dog <region1>?",
"Answer": "The person is walking the dog."
}
```
Another Example:
Input:
[Objects]: <region0> car, <region1> building. [Description]: The <region0> is parked in front of the <region1>.
Output:
```json
{
"Question": "Where is the car <region0> in relation to the building <region1>?",
"Answer": "The car is parked in front of the building."
}
```
Make sure your output is a valid JSON object.
"""


class GeneralizedSceneGraphGenerator:

    def __init__(self, config_path="config/v2_hf_qwen.py", device="cpu",
                 llm_model_name_hf=None, llm_device_hf="cpu"):

        self.logger = setup_logger(name="GeneralizedSceneGraphGenerator")
        init_start_time = time.perf_counter()
        self.logger.info(
            f"--- Initializing GeneralizedSceneGraphGenerator (Config: {config_path}, Device: {device}, LLM: {llm_model_name_hf} on {llm_device_hf}) ---")

        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.cfg = Config.fromfile(config_path)
        self.device = device
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        if not OSDSYNTH_AVAILABLE:
            self.logger.warning("osdsynth library components are not available. Some functionalities will be dummied.")

        yoloe_init_start = time.perf_counter()
        self.yoloe_model_path = self.cfg.get("yoloe_model_path", "yoloe-11l-seg-pf.pt")
        try:
            self.logger.info(f"Initializing YOLOE model from: {self.yoloe_model_path}")
            self.yoloe_model = YOLOE(self.yoloe_model_path)
            # self.yoloe_model.to(self.device) # YOLOE handles device internally based on availability
            self.logger.info(f"Successfully initialized YOLOE model (device auto-selected by YOLOE).")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOE model: {e}")
            raise RuntimeError("Could not initialize YOLOE model.")
        yoloe_init_end = time.perf_counter()
        self.logger.info(f"YOLOE model initialization took {yoloe_init_end - yoloe_init_start:.4f} seconds.")

        # DEPTH ANYTHING V2 INITIALIZATION
        depth_model_init_start = time.perf_counter()
        self.depth_model = None
        if not DEPTH_ANYTHING_V2_AVAILABLE:
            self.logger.error("DepthAnythingV2 library is not available. Cannot proceed with depth estimation.")
            raise ImportError("DepthAnythingV2 library failed to import.")

        self.logger.info("Initializing Depth Anything V2 model...")
        try:
            self.depth_anything_encoder = self.cfg.get("depth_anything_encoder", "vitl")
            self.depth_anything_load_from = self.cfg.get("depth_anything_load_from")

            if not self.depth_anything_load_from or not os.path.exists(self.depth_anything_load_from):
                raise ValueError(
                    f"Path to Depth Anything V2 model weights 'depth_anything_load_from' "
                    f"not found or not configured: {self.depth_anything_load_from}"
                )

            self.depth_anything_max_depth = self.cfg.get("depth_anything_max_depth", 20.0)
            self.depth_model_input_size = self.cfg.get("depth_model_input_size", 518)  # Default from DA2

            _model_configs_da = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            if self.depth_anything_encoder not in _model_configs_da:
                raise ValueError(
                    f"Unsupported Depth Anything V2 encoder: {self.depth_anything_encoder}. "
                    f"Supported: {list(_model_configs_da.keys())}"
                )

            model_params = _model_configs_da[self.depth_anything_encoder]
            # Assuming official DepthAnythingV2 constructor:
            # DepthAnythingV2(encoder='vitl', features=256, out_channels=[...], use_bn=False, use_clstoken=False, max_depth=20)
            # Add other params like 'use_bn', 'use_clstoken' if they are in cfg or required by your specific DA2 version
            self.depth_model = DepthAnythingV2(**{**model_params, 'max_depth': self.depth_anything_max_depth})

            self.depth_model.load_state_dict(torch.load(self.depth_anything_load_from, map_location='cpu'))
            self.depth_model = self.depth_model.to(self.device).eval()
            self.logger.info(
                f"Successfully initialized Depth Anything V2 model ({self.depth_anything_encoder}) on {self.device}.")

        except Exception as e:
            self.logger.error(f"Failed to initialize Depth Anything V2 model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize Depth Anything V2 model: {e}")
        depth_model_init_end = time.perf_counter()
        self.logger.info(
            f"Depth Anything V2 model initialization took {depth_model_init_end - depth_model_init_start:.4f} seconds.")

        osds_init_start = time.perf_counter()
        self.captioner = CaptionImage(self.cfg, self.logger, self.device, init_lava=False)
        self.qa_prompter = QAPromptGenerator(self.cfg, self.logger, self.device)
        self.fact_prompter = FactPromptGenerator(self.cfg, self.logger, self.device)
        osds_init_end = time.perf_counter()
        self.logger.info(
            f"OSDSynth components (Captioner, Prompters) initialization took {osds_init_end - osds_init_start:.4f} seconds.")

        self.llm_pipeline = None
        self.llm_tokenizer_hf = None
        if llm_model_name_hf and HF_TRANSFORMERS_AVAILABLE:
            llm_init_start = time.perf_counter()
            self.logger.info(f"Initializing Hugging Face LLM for: {llm_model_name_hf} on device: {llm_device_hf}")
            try:
                self.llm_tokenizer_hf = AutoTokenizer.from_pretrained(llm_model_name_hf, trust_remote_code=True)
                if self.llm_tokenizer_hf.pad_token is None:
                    if self.llm_tokenizer_hf.eos_token:
                        self.llm_tokenizer_hf.pad_token = self.llm_tokenizer_hf.eos_token
                    elif self.llm_tokenizer_hf.unk_token:  # Fallback for pad_token
                        self.llm_tokenizer_hf.pad_token = self.llm_tokenizer_hf.unk_token
                        self.logger.warning("LLM tokenizer missing eos_token, using unk_token as pad_token.")
                    else:  # Critical fallback: Add a pad token if none exists
                        self.llm_tokenizer_hf.add_special_tokens({'pad_token': '[PAD]'})
                        self.logger.warning("LLM tokenizer missing eos and unk tokens. Added new [PAD] token.")

                self.llm_pipeline = pipeline("text-generation", model=llm_model_name_hf,
                                             tokenizer=self.llm_tokenizer_hf,
                                             device_map=llm_device_hf if llm_device_hf in ["cpu", "auto"] else None,
                                             # device_map or device
                                             device=None if llm_device_hf == "auto" else llm_device_hf,
                                             # device arg if not auto and not multi-GPU
                                             torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
                                             trust_remote_code=True)
                # Ensure model is on CPU if device_map is "cpu" or device is "cpu"
                if hasattr(self.llm_pipeline, 'model') and (
                        (isinstance(llm_device_hf, str) and llm_device_hf == "cpu") or \
                        (isinstance(self.llm_pipeline.device,
                                    torch.device) and self.llm_pipeline.device.type == 'cpu')):
                    self.llm_pipeline.model.to("cpu")

                self.logger.info(
                    f"Hugging Face LLM pipeline for {llm_model_name_hf} initialized effectively on {self.llm_pipeline.device if hasattr(self.llm_pipeline, 'device') else llm_device_hf}.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Hugging Face LLM pipeline: {e}", exc_info=True)
                self.llm_pipeline = None;
                self.llm_tokenizer_hf = None
            llm_init_end = time.perf_counter()
            self.logger.info(
                f"Hugging Face LLM ({llm_model_name_hf}) initialization took {llm_init_end - llm_init_start:.4f} seconds.")

        elif llm_model_name_hf and not HF_TRANSFORMERS_AVAILABLE:
            self.logger.warning("llm_model_name_hf provided, but Hugging Face Transformers is not installed.")

        default_wis3d_folder = os.path.join(self.cfg.get("log_dir", "./temp_outputs/log"),
                                            f"Wis3D_Generalized_HF_{self.timestamp}")
        self.cfg.wis3d_folder = self.cfg.get("wis3d_folder", default_wis3d_folder)
        os.makedirs(self.cfg.wis3d_folder, exist_ok=True)
        self.cfg.vis = self.cfg.get("vis", False)

        init_end_time = time.perf_counter()
        self.logger.info(
            f"--- Total GeneralizedSceneGraphGenerator initialization took {init_end_time - init_start_time:.4f} seconds ---")

    def _override_config_and_reinit(self, **kwargs):
        reinit_captioner = False;
        reinit_depth_model = False
        for key, value in kwargs.items():
            parts = key.split('.');
            cfg_node = self.cfg;
            changed = False
            try:
                for i, part in enumerate(parts[:-1]): cfg_node = cfg_node[part]
                if cfg_node.get(parts[-1]) != value: cfg_node[parts[-1]] = value; changed = True
            except KeyError:
                cfg_node[parts[-1]] = value; changed = True  # Add if not exist

            if changed:
                self.logger.info(f"Config overridden: {key} = {value}")
                if key.startswith("llava_") or key.startswith("global_qs_list"):
                    reinit_captioner = True
                if key.startswith("depth_anything_") or key.startswith("depth_model_input_size"):
                    reinit_depth_model = True

        if reinit_captioner:
            reinit_cap_start = time.perf_counter()
            # Preserve LLaVA state if it was initialized
            init_lava_flag = hasattr(self.captioner, 'llava_processor') and self.captioner.llava_processor is not None
            self.captioner = CaptionImage(self.cfg, self.logger, self.device, init_lava=init_lava_flag)
            reinit_cap_end = time.perf_counter()
            self.logger.info(f"Captioner re-initialization took {reinit_cap_end - reinit_cap_start:.4f} seconds.")

        if reinit_depth_model and DEPTH_ANYTHING_V2_AVAILABLE:
            reinit_depth_start = time.perf_counter()
            self.logger.info("Re-initializing Depth Anything V2 model due to config changes...")
            try:
                # Simplified re-init logic, assuming full re-instantiation
                self.depth_anything_encoder = self.cfg.get("depth_anything_encoder", "vitl")
                self.depth_anything_load_from = self.cfg.get("depth_anything_load_from")
                if not self.depth_anything_load_from or not os.path.exists(self.depth_anything_load_from):
                    raise ValueError(
                        f"Path for re-init 'depth_anything_load_from' not found: {self.depth_anything_load_from}")
                self.depth_anything_max_depth = self.cfg.get("depth_anything_max_depth", 20.0)
                self.depth_model_input_size = self.cfg.get("depth_model_input_size", 518)

                _model_configs_da = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                model_params = _model_configs_da[self.depth_anything_encoder]

                if hasattr(self, 'depth_model') and self.depth_model is not None: del self.depth_model
                self.depth_model = DepthAnythingV2(**{**model_params, 'max_depth': self.depth_anything_max_depth})
                self.depth_model.load_state_dict(torch.load(self.depth_anything_load_from, map_location='cpu'))
                self.depth_model = self.depth_model.to(self.device).eval()
                self.logger.info(
                    f"Successfully re-initialized Depth Anything V2 model ({self.depth_anything_encoder}).")
            except Exception as e:
                self.logger.error(f"Failed to re-initialize Depth Anything V2 model: {e}", exc_info=True)
                # Decide: raise error or continue with old/no model? For now, log and potentially disable.
                self.depth_model = None
            reinit_depth_end = time.perf_counter()
            self.logger.info(
                f"Depth Anything V2 model re-initialization took {reinit_depth_end - reinit_depth_start:.4f} seconds.")

    def _load_image(self, image_input):
        load_img_start = time.perf_counter()
        if isinstance(image_input, str):
            if not os.path.exists(image_input): raise FileNotFoundError(f"Image not found at {image_input}")
            image_bgr = cv2.imread(image_input)
            if image_bgr is None: raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            image_bgr = image_input.copy()  # Ensure it's BGR
        else:
            raise TypeError("image_input must be a file path (str) or a NumPy array (BGR).")

        h_orig, w_orig = image_bgr.shape[:2]
        if h_orig == 0 or w_orig == 0: raise ValueError("Image has zero height or width.")

        target_h = self.cfg.get("image_resize_height", 512)  # DA2 default is 518 for input_size (shorter edge)
        # Using image_resize_height for consistency with UniK3D setup.
        # This height is for the image passed to YOLOE and DA2.
        if target_h <= 0:  # Use original height if target_h is invalid or 0/None
            scale = 1.0
            target_w = w_orig
            image_bgr_resized = image_bgr.copy()
            self.logger.info(f"  Using original image size: {w_orig}x{h_orig}")
        else:
            scale = target_h / h_orig
            target_w = int(w_orig * scale)
            image_bgr_resized = cv2.resize(image_bgr, (target_w, target_h),
                                           interpolation=cv2.INTER_LANCZOS4 if scale < 1 else cv2.INTER_LINEAR)
            self.logger.info(f"  Image resized from {w_orig}x{h_orig} to {target_w}x{target_h} (scale: {scale:.3f})")

        load_img_end = time.perf_counter()
        self.logger.info(f"  _load_image (load & resize) took {load_img_end - load_img_start:.4f} seconds.")
        return image_bgr_resized  # This is the image used for subsequent processing (segmentation, depth)

    def _get_object_classes(self, image_rgb_pil, custom_vocabulary=None):
        get_cls_start = time.perf_counter()
        yoloe_class_names = list(self.yoloe_model.names.values())
        result_classes = yoloe_class_names

        if custom_vocabulary:
            if not isinstance(custom_vocabulary, list) or not all(isinstance(s, str) for s in custom_vocabulary):
                raise ValueError("custom_vocabulary must be a list of strings.")
            if not custom_vocabulary:  # Allow empty list to mean "detect all YOLOE classes"
                self.logger.info("Empty custom_vocabulary provided, detecting all YOLOE classes.")
            else:  # Non-empty custom vocab
                valid_custom_classes = [cls for cls in custom_vocabulary if cls in yoloe_class_names]
                if not valid_custom_classes:
                    self.logger.warning(
                        f"None of the custom vocabulary classes {custom_vocabulary} are detectable by YOLOE. Available: {yoloe_class_names}. Detecting all YOLOE classes instead.")
                else:
                    self.logger.info(
                        f"Using custom vocabulary (filtered for YOLOE compatibility): {valid_custom_classes}")
                    result_classes = valid_custom_classes
        else:  # custom_vocabulary is None
            self.logger.info(f"Using all detectable classes from YOLOE: {yoloe_class_names}")

        get_cls_end = time.perf_counter()
        self.logger.info(
            f"  _get_object_classes took {get_cls_end - get_cls_start:.4f} seconds. Classes to detect: {result_classes if result_classes else 'All'}")
        return result_classes

    def _segment_image(self, image_bgr, classes_to_detect):  # image_bgr is already resized
        segment_total_start = time.perf_counter()
        self.logger.info("  Starting YOLOE segmentation...")
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # YOLOE predict can take BGR
        image_rgb_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))  # For cropping

        target_class_indices = None
        if classes_to_detect:  # classes_to_detect is a list of names
            name_to_idx_map = {name: idx for idx, name in self.yoloe_model.names.items()}
            target_class_indices = [name_to_idx_map[name] for name in classes_to_detect if name in name_to_idx_map]
            if not target_class_indices:  # If filtering results in empty list, means no specified classes are in YOLOE
                self.logger.warning(
                    f"No valid class indices for YOLOE from classes_to_detect: {classes_to_detect}. YOLOE will detect all its classes.")
                # Let YOLOE detect all if filtering yields nothing. Or, could raise SkipImageException.

        yolo_predict_start = time.perf_counter()
        # YOLOE's predict method takes source (path, PIL, np.ndarray BGR/RGB)
        # It internally handles conversion if needed.
        yolo_results = self.yoloe_model.predict(
            source=image_bgr,  # Pass the BGR numpy array
            classes=target_class_indices if target_class_indices else None,  # None means all classes
            conf=self.cfg.get("yoloe_confidence_threshold", 0.6)
        )
        yolo_predict_end = time.perf_counter()
        self.logger.info(f"    YOLOE model.predict() took {yolo_predict_end - yolo_predict_start:.4f} seconds.")

        if not yolo_results or not yolo_results[0].boxes or yolo_results[0].boxes.shape[0] == 0:
            raise SkipImageException(f"No objects detected by YOLOE for specified classes/confidence.")

        res = yolo_results[0]  # Results for the first (and only) image

        if res.masks is None or res.masks.data is None or res.masks.data.shape[0] == 0:
            raise SkipImageException(f"No masks found in YOLOE results, though boxes were detected.")

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confidences = res.boxes.conf.cpu().numpy()
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        # Masks from YOLOE are typically [N, H, W] and correspond to original image dimensions if not resized by predict call
        # YOLOE's masks.data are usually normalized to image input size of predict.
        masks_data_np = res.masks.data.cpu().numpy()

        mask_proc_start = time.perf_counter()
        h_img, w_img = image_bgr.shape[:2]  # Dimensions of the image YOLOE processed
        processed_masks = []
        for i in range(masks_data_np.shape[0]):  # Iterate over N masks
            mask_i = masks_data_np[i, :, :]  # H, W
            # Ensure mask_i has same dimensions as image_bgr
            if mask_i.shape[0] != h_img or mask_i.shape[1] != w_img:
                mask_i = cv2.resize(mask_i, (w_img, h_img),
                                    interpolation=cv2.INTER_LINEAR)  # INTER_NEAREST for binary masks usually
            binary_mask = (mask_i > self.cfg.get("yoloe_mask_threshold", 0.5)).astype(bool)
            processed_masks.append(binary_mask)
        mask_proc_end = time.perf_counter()
        self.logger.info(
            f"    YOLOE mask data processing (resizing, binarizing) took {mask_proc_end - mask_proc_start:.4f} seconds.")

        if not processed_masks:
            raise SkipImageException("No masks could be processed from YOLOE output.")
        masks_np_stack = np.stack(processed_masks)

        detected_class_names = [self.yoloe_model.names[cid] for cid in class_ids]

        det_list_form_start = time.perf_counter()
        detection_list = []
        for i in range(len(boxes_xyxy)):
            detection_list.append({
                "xyxy": boxes_xyxy[i],
                "mask": masks_np_stack[i],  # Full image binary mask
                "subtracted_mask": masks_np_stack[i].copy(),  # To be modified by mask subtraction later
                "confidence": confidences[i],
                "class_name": detected_class_names[i],
                "class_id": class_ids[i]
            })
        det_list_form_end = time.perf_counter()
        self.logger.info(
            f"    Formulating initial detection list took {det_list_form_end - det_list_form_start:.4f} seconds.")

        if not detection_list:
            raise SkipImageException("No detections formulated into list.")

        # Filter by mask area BEFORE sorting and other processing
        filter_area_start = time.perf_counter()
        filtered_detection_list_area = []
        min_area = self.cfg.get("min_mask_area_pixel", 100)
        for det in detection_list:
            mask_area = np.sum(det["mask"])
            if mask_area >= min_area:
                filtered_detection_list_area.append(det)
            else:
                self.logger.debug(
                    f"Filtering out {det['class_name']} (conf: {det['confidence']:.2f}) due to small mask area: {mask_area} < {min_area}")
        filter_area_end = time.perf_counter()
        self.logger.info(
            f"    Filtering detections by mask area took {filter_area_end - filter_area_start:.4f} seconds. Kept {len(filtered_detection_list_area)} of {len(detection_list)} detections.")

        if not filtered_detection_list_area:
            raise SkipImageException("No detections remaining after area filtering.")
        detection_list = filtered_detection_list_area  # Update list

        # Sort by area (largest first) - useful for mask subtraction if enabled
        sort_area_start = time.perf_counter()
        detection_list = sorted(detection_list, key=lambda d: np.sum(d["mask"]), reverse=True)
        sort_area_end = time.perf_counter()
        self.logger.info(f"    Sorting detections by area took {sort_area_end - sort_area_start:.4f} seconds.")

        # Mask dilation (optional)
        if self.cfg.get("yoloe_mask_dilate_iterations", 0) > 0:
            dilate_start = time.perf_counter()
            kernel_size = self.cfg.get("yoloe_mask_dilate_kernel_size", 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            iterations = self.cfg.get("yoloe_mask_dilate_iterations")
            for det in detection_list:
                mask_uint8 = det["mask"].astype(np.uint8)  # Dilate needs uint8
                dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
                det["mask"] = dilated_mask.astype(bool)  # Back to boolean
                det["subtracted_mask"] = dilated_mask.astype(bool)  # Update this too
            dilate_end = time.perf_counter()
            self.logger.info(f"    Mask dilation took {dilate_end - dilate_start:.4f} seconds.")

        # Cropping (uses PIL image and full masks)
        crop_start = time.perf_counter()
        # Ensure the globally defined crop_detections_with_xyxy is used if osdsynth one isn't available
        _crop_fn = crop_detections_with_xyxy
        if OSDSYNTH_AVAILABLE and hasattr(sys.modules['osdsynth.processor.wrappers.sam'], 'crop_detections_with_xyxy'):
            _crop_fn = sys.modules['osdsynth.processor.wrappers.sam'].crop_detections_with_xyxy

        detection_list = _crop_fn(self.cfg, image_rgb_pil, detection_list)
        crop_end = time.perf_counter()
        self.logger.info(f"    Cropping detections took {crop_end - crop_start:.4f} seconds.")

        segment_total_end = time.perf_counter()
        self.logger.info(
            f"  _segment_image (YOLOE) total took {segment_total_end - segment_total_start:.4f} seconds. Found {len(detection_list)} detections.")
        return detection_list

    def _process_common(self, image_input, custom_vocabulary=None, **kwargs):
        process_common_total_start = time.perf_counter()
        image_name_for_log = image_input if isinstance(image_input, str) else 'numpy_array_input'
        self.logger.info(
            f"--- Starting _process_common for image: {os.path.basename(image_name_for_log) if isinstance(image_input, str) else image_name_for_log} ---")

        override_start = time.perf_counter()
        self._override_config_and_reinit(**kwargs)
        override_end = time.perf_counter()
        self.logger.info(f"  _override_config_and_reinit took {override_end - override_start:.4f} seconds.")

        image_bgr = self._load_image(image_input)  # This is the BGR image, resized.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # RGB version for colors, some models

        filename_prefix = "processed_image_" + self.timestamp
        if isinstance(image_input, str):
            filename_prefix = os.path.splitext(os.path.basename(image_input))[0] + "_" + self.timestamp

        # Segmentation uses image_bgr (resized)
        object_classes_to_detect = self._get_object_classes(Image.fromarray(image_rgb), custom_vocabulary)
        detection_list_initial_dicts = self._segment_image(image_bgr, object_classes_to_detect)

        if not detection_list_initial_dicts:
            raise SkipImageException("Segmentation (YOLOE) resulted in no initial detections.")

        # DEPTH ANYTHING V2 INFERENCE
        depth_infer_start = time.perf_counter()
        if not self.depth_model:
            raise RuntimeError("Depth model (Depth Anything V2) is not initialized.")

        h_proc, w_proc = image_bgr.shape[:2]  # Dimensions of the image fed to Depth Anything
        with torch.no_grad():
            # Depth Anything V2's infer_image takes a BGR numpy array.
            # The `input_size` parameter in `infer_image` refers to the shorter edge for internal resizing.
            # The model's `infer_image` is expected to return a depth map resized to original input's H, W.
            depth_map_np = self.depth_model.infer_image(image_bgr, input_size=self.depth_model_input_size)
            # depth_map_np should now have shape (h_proc, w_proc) and metric depth values.

        depth_infer_end = time.perf_counter()
        self.logger.info(
            f"  Depth Anything V2 inference took {depth_infer_end - depth_infer_start:.4f} seconds. Output depth map shape: {depth_map_np.shape}")

        # CONVERT DEPTH MAP TO POINT CLOUD
        pc_conversion_start = time.perf_counter()

        # Focal lengths - ensure they correspond to the processed image dimensions (h_proc, w_proc)
        fx_config = self.cfg.get("depth_anything_focal_length_x", None)
        fy_config = self.cfg.get("depth_anything_focal_length_y", None)

        if fx_config is not None and isinstance(fx_config, (int, float)) and fx_config > 0:
            fx = fx_config
        else:
            hfov_degrees = self.cfg.get("camera_hfov_degrees", 60)
            fx = (w_proc / 2.0) / tan(radians(hfov_degrees / 2.0)) if tan(
                radians(hfov_degrees / 2.0)) != 0 else w_proc  # Avoid div by zero
            self.logger.info(f"  Using calculated fx={fx:.2f} based on w_proc={w_proc} and HFOV={hfov_degrees} deg.")

        if fy_config is not None and isinstance(fy_config, (int, float)) and fy_config > 0:
            fy = fy_config
        else:
            vfov_degrees = self.cfg.get("camera_vfov_degrees", 45)
            fy = (h_proc / 2.0) / tan(radians(vfov_degrees / 2.0)) if tan(
                radians(vfov_degrees / 2.0)) != 0 else h_proc  # Avoid div by zero
            self.logger.info(f"  Using calculated fy={fy:.2f} based on h_proc={h_proc} and VFOV={vfov_degrees} deg.")

        cx = w_proc / 2.0
        cy = h_proc / 2.0

        x_coords, y_coords = np.meshgrid(np.arange(w_proc), np.arange(h_proc), indexing='xy')

        # Filter out zero or negative depth values if they are not meaningful (e.g., background)
        # For DA2, depth is metric and positive. Small values are near, large are far (up to max_depth).
        # Points where depth_map_np <= 0 might be invalid, consider masking them out or setting Z to a large value.
        # For now, use raw depth values.
        valid_depth_mask = depth_map_np > self.cfg.get("min_valid_depth",
                                                       0.01)  # Points too close or at zero depth might be problematic

        points_x = np.full_like(depth_map_np, np.nan)
        points_y = np.full_like(depth_map_np, np.nan)
        points_z = np.full_like(depth_map_np, np.nan)

        points_x[valid_depth_mask] = (x_coords[valid_depth_mask] - cx) * depth_map_np[valid_depth_mask] / fx
        points_y[valid_depth_mask] = (y_coords[valid_depth_mask] - cy) * depth_map_np[
            valid_depth_mask] / fy  # Y positive downwards (camera convention)
        # To make Y positive upwards (common 3D convention), use:
        # points_y[valid_depth_mask] = -(y_coords[valid_depth_mask] - cy) * depth_map_np[valid_depth_mask] / fy
        points_z[valid_depth_mask] = depth_map_np[valid_depth_mask]

        points_3d_global = np.stack((points_x, points_y, points_z), axis=-1)  # Shape: (h_proc, w_proc, 3)

        pc_conversion_end = time.perf_counter()
        self.logger.info(
            f"  Depth map to Point Cloud conversion took {pc_conversion_end - pc_conversion_start:.4f} seconds.")

        wis3d_instance = None
        if self.cfg.get("vis", False):
            wis3d_setup_start = time.perf_counter()
            wis3d_instance = Wis3D(self.cfg.wis3d_folder, filename_prefix)
            # Reshape for Wis3D: from (H, W, 3) to (N, 3)
            global_points_flat = points_3d_global.reshape(-1, 3)
            global_colors_flat = image_rgb.reshape(-1, 3) / 255.0  # Normalize colors

            # Filter out NaN points for visualization
            nan_mask_flat = ~np.isnan(global_points_flat).any(axis=1)
            valid_global_points = global_points_flat[nan_mask_flat]
            valid_global_colors = global_colors_flat[nan_mask_flat]

            if valid_global_points.shape[0] > 0:
                wis3d_instance.add_point_cloud(
                    vertices=valid_global_points,
                    colors=valid_global_colors,
                    name="depth_anything_v2_global_scene_pts")
            else:
                self.logger.warning("No valid global points to visualize in Wis3D after NaN filtering.")
            wis3d_setup_end = time.perf_counter()
            self.logger.info(
                f"  Wis3D setup and global cloud add took {wis3d_setup_end - wis3d_setup_start:.4f} seconds.")

        obj_proc_loop_start = time.perf_counter()
        valid_detections_dicts = []
        min_initial_pts = self.cfg.get("min_points_threshold", 20)
        min_processed_pts = self.cfg.get("min_points_threshold_after_denoise", 10)
        min_bbox_volume = self.cfg.get("bbox_min_volume_threshold", 1e-6)  # meters^3
        num_initial_dets = len(detection_list_initial_dicts)

        for det_idx, det_dict in enumerate(detection_list_initial_dicts):
            per_obj_proc_start = time.perf_counter()
            object_mask = det_dict["subtracted_mask"]  # This is a 2D boolean mask (H, W)
            if not np.any(object_mask):
                self.logger.debug(
                    f"    Skipping det {det_idx + 1}/{num_initial_dets} ({det_dict.get('class_name', 'N/A')}) due to empty mask.")
                continue

            # Extract points and colors for this object using the 2D mask
            obj_points_from_global = points_3d_global[object_mask]  # Selects (N_obj_pixels, 3)
            obj_colors_from_global = image_rgb[object_mask] / 255.0  # Normalize colors

            # Filter out NaN points that fell within the mask
            nan_mask_obj = ~np.isnan(obj_points_from_global).any(axis=1)
            obj_points_valid = obj_points_from_global[nan_mask_obj]
            obj_colors_valid = obj_colors_from_global[nan_mask_obj]

            if len(obj_points_valid) < min_initial_pts:
                self.logger.debug(
                    f"    Skipping det {det_idx + 1}/{num_initial_dets} ({det_dict.get('class_name', 'N/A')}): "
                    f"Too few initial valid points ({len(obj_points_valid)} < {min_initial_pts}).")
                continue

            # Optional: Add small perturbation (if needed for robustness of downstream, e.g. OBB)
            # obj_points_valid += np.random.normal(0, self.cfg.get("pcd_perturb_std", 1e-4), obj_points_valid.shape) # Smaller perturbation

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points_valid)
            pcd.colors = o3d.utility.Vector3dVector(obj_colors_valid)

            pcd_processing_start = time.perf_counter()
            processed_pcd = process_pcd_generic(self.cfg, pcd)  # Using renamed generic helper
            pcd_processing_end = time.perf_counter()

            if not processed_pcd.has_points() or len(processed_pcd.points) < min_processed_pts:
                self.logger.debug(
                    f"    Skipping det {det_idx + 1}/{num_initial_dets} ({det_dict.get('class_name', 'N/A')}): "
                    f"Too few points after processing ({len(processed_pcd.points)} < {min_processed_pts}).")
                continue

            bbox_calc_start = time.perf_counter()
            axis_aligned_bbox, oriented_bbox = get_bounding_box_generic(self.cfg, processed_pcd)  # Using renamed
            bbox_calc_end = time.perf_counter()

            if axis_aligned_bbox.is_empty() or axis_aligned_bbox.volume() < min_bbox_volume:
                self.logger.debug(
                    f"    Skipping det {det_idx + 1}/{num_initial_dets} ({det_dict.get('class_name', 'N/A')}): "
                    f"BBox volume too small ({axis_aligned_bbox.volume()}).")
                continue

            det_dict["pcd"] = processed_pcd
            det_dict["axis_aligned_bbox"] = axis_aligned_bbox
            det_dict["oriented_bbox"] = oriented_bbox
            valid_detections_dicts.append(det_dict)
            per_obj_proc_end = time.perf_counter()
            self.logger.debug(
                f"    Processing detection {det_idx + 1}/{num_initial_dets} ({det_dict.get('class_name', 'N/A')}): "
                f"PCD processing: {pcd_processing_end - pcd_processing_start:.4f}s, "
                f"BBox calc: {bbox_calc_end - bbox_calc_start:.4f}s, "
                f"Total for object: {per_obj_proc_end - per_obj_proc_start:.4f}s.")
        obj_proc_loop_end = time.perf_counter()
        self.logger.info(
            f"  3D object processing loop for {len(valid_detections_dicts)} valid objects (out of {num_initial_dets} initial) took {obj_proc_loop_end - obj_proc_loop_start:.4f} seconds.")

        if not valid_detections_dicts:
            raise SkipImageException("No valid 3D objects found after processing.")

        if wis3d_instance:
            wis3d_obj_vis_start = time.perf_counter()
            object_pcds_for_vis = [d["pcd"] for d in valid_detections_dicts]
            instance_colored_pcds = color_by_instance_generic(object_pcds_for_vis)  # Using renamed
            for i, det_data_dict in enumerate(valid_detections_dicts):
                obj_pcd_colored = instance_colored_pcds[i]
                class_name = det_data_dict.get("class_name", f"object_{i}")
                if obj_pcd_colored.has_points():
                    wis3d_instance.add_point_cloud(vertices=np.asarray(obj_pcd_colored.points),
                                                   colors=np.asarray(obj_pcd_colored.colors),
                                                   name=f"{i:02d}_{class_name}_obj_pts")
                aa_bbox = det_data_dict["axis_aligned_bbox"]
                if not aa_bbox.is_empty():
                    aa_center, aa_eulers, aa_extent = axis_aligned_bbox_to_center_euler_extent_generic(  # Renamed
                        aa_bbox.get_min_bound(), aa_bbox.get_max_bound())
                    wis3d_instance.add_boxes(positions=np.array([aa_center]), eulers=np.array([aa_eulers]),
                                             extents=np.array([aa_extent]), name=f"{i:02d}_{class_name}_aa_bbox")
                or_bbox = det_data_dict["oriented_bbox"]
                # Check extent directly as is_empty might not be enough for degenerate OBBs
                if not or_bbox.is_empty() and np.all(np.array(or_bbox.extent) > 1e-6):  # Check extent for valid OBB
                    or_center, or_eulers, or_extent = oriented_bbox_to_center_euler_extent_generic(  # Renamed
                        or_bbox.center, or_bbox.R, or_bbox.extent)
                    wis3d_instance.add_boxes(positions=np.array([or_center]), eulers=np.array([or_eulers]),
                                             extents=np.array([or_extent]), name=f"{i:02d}_{class_name}_or_bbox")
            wis3d_obj_vis_end = time.perf_counter()
            self.logger.info(
                f"  Wis3D visualization for {len(valid_detections_dicts)} individual objects took {wis3d_obj_vis_end - wis3d_obj_vis_start:.4f} seconds.")

        captioner_start = time.perf_counter()
        # Ensure captioner gets necessary data; "image_crop" should be PIL Image, "mask_crop" np.ndarray
        captioned_detections_dicts = self.captioner.process_local_caption(valid_detections_dicts)
        captioner_end = time.perf_counter()
        self.logger.info(
            f"  Captioner (process_local_caption) took {captioner_end - captioner_start:.4f} seconds for {len(valid_detections_dicts)} objects.")

        det_obj_inst_start = time.perf_counter()
        detected_object_instances = []
        for det_dict in captioned_detections_dicts:
            description = det_dict.get("caption", det_dict.get("class_name", "Unknown Object"))
            pcd_o3d = det_dict.get("pcd");
            obb_o3d = det_dict.get("oriented_bbox");
            aabb_o3d = det_dict.get("axis_aligned_bbox")
            # Use "mask" (full image mask) for DetectedObject, not "mask_crop" or "subtracted_mask" unless intended
            mask_2d_np = det_dict.get("mask")  # Full mask from YOLOE
            bbox_2d_np = det_dict.get("xyxy")

            if not all([pcd_o3d, obb_o3d, aabb_o3d, mask_2d_np is not None, bbox_2d_np is not None]):
                self.logger.warning(
                    f"Skipping object '{det_dict.get('class_name', 'N/A')}' due to missing critical data (PCD, BBoxes, 2D Mask, 2D BBox) for DetectedObject.")
                continue

            obj_instance = DetectedObject(class_name=det_dict.get("class_name", "Unknown"), description=description,
                                          segmentation_mask_2d=mask_2d_np, bounding_box_2d=bbox_2d_np,
                                          point_cloud_3d=pcd_o3d, bounding_box_3d_oriented=obb_o3d,
                                          bounding_box_3d_axis_aligned=aabb_o3d,
                                          image_crop_pil=det_dict.get("image_crop"))
            detected_object_instances.append(obj_instance)
        det_obj_inst_end = time.perf_counter()
        self.logger.info(
            f"  Creating DetectedObject instances took {det_obj_inst_end - det_obj_inst_start:.4f} seconds for {len(detected_object_instances)} objects.")

        if not detected_object_instances:
            raise SkipImageException("No valid DetectedObject instances could be created.")

        process_common_total_end = time.perf_counter()
        self.logger.info(
            f"--- _process_common total took {process_common_total_end - process_common_total_start:.4f} seconds. Produced {len(detected_object_instances)} DetectedObjects. ---")
        return detected_object_instances, captioned_detections_dicts, filename_prefix

    def generate_facts(self, image_input, custom_vocabulary=None, run_llm_rephrase=False, **kwargs):
        gen_facts_total_start = time.perf_counter()
        image_name_for_log = image_input if isinstance(image_input, str) else 'numpy_array_input'
        self.logger.info(
            f"--- Starting generate_facts for image: {os.path.basename(image_name_for_log) if isinstance(image_input, str) else image_name_for_log} ---")
        try:
            proc_common_call_start = time.perf_counter()
            detected_objects_list, detection_list_dicts, filename_prefix = self._process_common(
                image_input, custom_vocabulary, **kwargs)
            proc_common_call_end = time.perf_counter()
            self.logger.info(
                f"  Call to _process_common (from generate_facts) took {proc_common_call_end - proc_common_call_start:.4f} seconds.")

            if not detected_objects_list:
                self.logger.warning("Common processing failed to produce detections for fact generation.")
                return [], [], []  # Return empty lists as per original behavior

            fact_prompter_start = time.perf_counter()
            template_facts = self.fact_prompter.evaluate_predicates_on_pairs(detection_list_dicts)
            fact_prompter_end = time.perf_counter()
            self.logger.info(
                f"  Fact prompter (evaluate_predicates_on_pairs) took {fact_prompter_end - fact_prompter_start:.4f} seconds, generated {len(template_facts)} facts.")

            rephrased_qas = []
            if run_llm_rephrase and template_facts:
                llm_rephrase_call_start = time.perf_counter()
                if not self.llm_pipeline:
                    self.logger.warning("LLM pipeline not initialized. Skipping LLM rephrasing for facts.")
                else:
                    prep_prompts_start = time.perf_counter()
                    llm_prompts = prepare_llm_prompts_from_facts(template_facts, detection_list_dicts)
                    prep_prompts_end = time.perf_counter()
                    self.logger.info(
                        f"    Preparing LLM prompts from facts took {prep_prompts_end - prep_prompts_start:.4f} seconds for {len(llm_prompts)} prompts.")

                    if llm_prompts:
                        rephrased_qas = self._run_llm_rephrasing_hf(llm_prompts)
                llm_rephrase_call_end = time.perf_counter()
                self.logger.info(
                    f"  LLM rephrasing section (for facts) took {llm_rephrase_call_end - llm_rephrase_call_start:.4f} seconds.")

            if self.cfg.get("vis", False) and filename_prefix: self.logger.info(
                f"Wis3D visualization potentially saved for {filename_prefix} in {self.cfg.wis3d_folder}")

            gen_facts_total_end = time.perf_counter()
            self.logger.info(
                f"--- generate_facts total took {gen_facts_total_end - gen_facts_total_start:.4f} seconds ---")
            return detected_objects_list, template_facts, rephrased_qas
        except SkipImageException as e:
            self.logger.warning(f"Fact generation skipped for image '{image_name_for_log}': {e}")
            return [], [], []
        except Exception as e:
            self.logger.error(f"Error during fact generation for image '{image_name_for_log}': {e}", exc_info=True)
            return [], [], []

    def generate_qa(self, image_input, custom_vocabulary=None, **kwargs):
        gen_qa_total_start = time.perf_counter()
        image_name_for_log = image_input if isinstance(image_input, str) else 'numpy_array_input'
        self.logger.info(
            f"--- Starting generate_qa for image: {os.path.basename(image_name_for_log) if isinstance(image_input, str) else image_name_for_log} ---")
        try:
            proc_common_call_start = time.perf_counter()
            detected_objects_list, detection_list_dicts, filename_prefix = self._process_common(
                image_input, custom_vocabulary, **kwargs)
            proc_common_call_end = time.perf_counter()
            self.logger.info(
                f"  Call to _process_common (from generate_qa) took {proc_common_call_end - proc_common_call_start:.4f} seconds.")

            if not detected_objects_list:
                self.logger.warning("Common processing failed to produce detections for QA generation.")
                return [], []

            qa_prompter_start = time.perf_counter()
            # VQA expects list of dicts with 'caption', 'class_name', 'image_crop' (PIL), 'mask_crop' (np.array)
            vqa_results = self.qa_prompter.evaluate_predicates_on_pairs(detection_list_dicts)
            qa_prompter_end = time.perf_counter()
            self.logger.info(
                f"  QA prompter (evaluate_predicates_on_pairs) took {qa_prompter_end - qa_prompter_start:.4f} seconds, produced {len(vqa_results)} VQA items.")

            parse_qa_start = time.perf_counter()
            template_qas = parse_qas_from_vqa_results(vqa_results)  # Expects list of items, each item has QA pair
            parse_qa_end = time.perf_counter()
            self.logger.info(
                f"  Parsing QAs from VQA results took {parse_qa_end - parse_qa_start:.4f} seconds, generated {len(template_qas)} QAs.")

            if self.cfg.get("vis", False) and filename_prefix: self.logger.info(
                f"Wis3D visualization potentially saved for {filename_prefix} in {self.cfg.wis3d_folder}")

            gen_qa_total_end = time.perf_counter()
            self.logger.info(f"--- generate_qa total took {gen_qa_total_end - gen_qa_total_start:.4f} seconds ---")
            return detected_objects_list, template_qas
        except SkipImageException as e:
            self.logger.warning(f"QA generation skipped for image '{image_name_for_log}': {e}")
            return [], []
        except Exception as e:
            self.logger.error(f"Error during QA generation for image '{image_name_for_log}': {e}", exc_info=True)
            return [], []

    def _parse_llm_json_output(self, llm_output_text):
        llm_output_text_stripped = llm_output_text.strip()
        # Try to find JSON block specifically delimited by ```json ... ```
        match_json_block = re.search(r"```json\s*(\{.*?\})\s*```", llm_output_text_stripped, re.DOTALL)
        json_str = None
        if match_json_block:
            json_str = match_json_block.group(1)
        else:
            # Fallback: find first '{' and last '}'
            first_brace = llm_output_text_stripped.find('{')
            last_brace = llm_output_text_stripped.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
                json_str = llm_output_text_stripped[first_brace: last_brace + 1]
            else:  # Simplest check: does it start and end with braces?
                if llm_output_text_stripped.startswith("{") and llm_output_text_stripped.endswith("}"):
                    json_str = llm_output_text_stripped

        if json_str is None:
            self.logger.warning(
                f"Could not clearly identify JSON block in LLM output: {llm_output_text_stripped[:200]}...")
            return None

        try:
            # Clean common escape issues before parsing
            json_str_cleaned = json_str.replace(r"\'", "'").replace(r'\"', '"')
            parsed_json = json.loads(json_str_cleaned)
            return parsed_json
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"JSONDecodeError for string (cleaned): '{json_str_cleaned[:200]}...'. Error: {e}. Original text (stripped): {llm_output_text_stripped[:200]}")
            return None

    def _run_llm_rephrasing_hf(self, llm_prompts):
        if not self.llm_pipeline or not self.llm_tokenizer_hf:
            self.logger.warning("Hugging Face LLM pipeline or tokenizer not available. Skipping rephrasing.")
            return []

        rephrased_conversations = []
        num_prompts_to_process = len(llm_prompts)
        if num_prompts_to_process == 0:
            self.logger.info("  No LLM prompts to rephrase.")
            return []

        self.logger.info(f"  Starting LLM rephrasing for {num_prompts_to_process} prompts.")
        total_llm_processing_time_for_all_prompts = 0

        for idx, user_prompt_text in enumerate(llm_prompts):
            single_prompt_overall_start_time = time.perf_counter()
            self.logger.info(
                f"    Processing LLM prompt {idx + 1}/{num_prompts_to_process}: '{user_prompt_text[:70]}...'")

            messages = [{"role": "system", "content": LLM_HF_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Input:\n{user_prompt_text}\nOutput:"}]
            try:
                template_apply_start = time.perf_counter()
                # For some models, add_generation_prompt=True is crucial. For others, it might add unwanted tokens.
                # Test with your specific LLM. Qwen2 usually benefits from it.
                prompt_for_llm = self.llm_tokenizer_hf.apply_chat_template(messages, tokenize=False,
                                                                           add_generation_prompt=True)
                template_apply_end = time.perf_counter()
                self.logger.debug(
                    f"      Applying chat template took {template_apply_end - template_apply_start:.4f}s.")
            except Exception as e_template:
                self.logger.error(
                    f"    Failed to apply chat template for prompt {idx + 1}: {e_template}. Skipping this prompt.",
                    exc_info=True)
                continue

            max_retries = self.cfg.get("llm_max_retries", 3);
            success = False;
            q_final, a_final = None, None
            for attempt in range(max_retries):
                llm_api_call_start = time.perf_counter()
                try:
                    terminators = []
                    if self.llm_tokenizer_hf.eos_token_id is not None: terminators.append(
                        self.llm_tokenizer_hf.eos_token_id)

                    # Check for model-specific terminators if tokenizer has them (e.g. Qwen <|im_end|>)
                    im_end_token_id = self.llm_tokenizer_hf.convert_tokens_to_ids("<|im_end|>")  # Qwen specific
                    if isinstance(im_end_token_id,
                                  int) and im_end_token_id not in terminators and im_end_token_id != self.llm_tokenizer_hf.unk_token_id:
                        terminators.append(im_end_token_id)

                    eot_id_llama = self.llm_tokenizer_hf.convert_tokens_to_ids("<|eot_id|>")  # Llama3 specific
                    if isinstance(eot_id_llama,
                                  int) and eot_id_llama not in terminators and eot_id_llama != self.llm_tokenizer_hf.unk_token_id:
                        terminators.append(eot_id_llama)

                    eos_pipeline_arg = terminators if terminators else None  # Use list of terminators or None

                    pipeline_args = {
                        "max_new_tokens": self.cfg.get("llm_max_new_tokens", 256),  # Reduced for QA
                        "temperature": self.cfg.get("llm_temperature", 0.1),
                        "do_sample": True if self.cfg.get("llm_temperature", 0.1) > 0 else False,
                        "top_p": self.cfg.get("llm_top_p", 0.95),
                        # "num_beams": self.cfg.get("llm_num_beams", 1) # Beam search can be slow
                    }
                    if eos_pipeline_arg: pipeline_args["eos_token_id"] = eos_pipeline_arg

                    # Pad token ID: Crucial for batching if ever used, good for single prompts too.
                    if self.llm_tokenizer_hf.pad_token_id is not None:
                        pipeline_args["pad_token_id"] = self.llm_tokenizer_hf.pad_token_id
                    elif self.llm_tokenizer_hf.eos_token_id is not None:  # Fallback to EOS if PAD is not set
                        pipeline_args["pad_token_id"] = self.llm_tokenizer_hf.eos_token_id

                    generated_outputs = self.llm_pipeline(prompt_for_llm, **pipeline_args)
                    llm_api_call_end = time.perf_counter()
                    self.logger.info(
                        f"      LLM pipeline call attempt {attempt + 1} for prompt {idx + 1} took {llm_api_call_end - llm_api_call_start:.4f} seconds.")

                    actual_llm_generation = None
                    if generated_outputs and isinstance(generated_outputs, list) and generated_outputs[0]:
                        if "generated_text" in generated_outputs[0]:
                            full_response_with_prompt = generated_outputs[0]["generated_text"]
                            if isinstance(full_response_with_prompt, str):
                                # Remove the input prompt part from the generated text
                                if full_response_with_prompt.startswith(prompt_for_llm):
                                    actual_llm_generation = full_response_with_prompt[len(prompt_for_llm):]
                                else:  # If prompt not found at start, maybe model didn't echo. Use full.
                                    self.logger.debug(
                                        "LLM output did not start with prompt. Using full output for parsing.")
                                    actual_llm_generation = full_response_with_prompt

                    if actual_llm_generation is not None and isinstance(actual_llm_generation, str):
                        # Clean known EOS tokens if they are part of the string but not handled by pipeline terminators
                        for known_eos in ["<|im_end|>", "<|eot_id|>", self.llm_tokenizer_hf.eos_token]:
                            if known_eos and actual_llm_generation.strip().endswith(known_eos):
                                actual_llm_generation = actual_llm_generation.strip()[:-len(known_eos)].strip()

                        json_response = self._parse_llm_json_output(actual_llm_generation)
                        if not json_response:
                            self.logger.warning(
                                f"      LLM attempt {attempt + 1} (prompt {idx + 1}): Could not parse JSON from: {actual_llm_generation[:200]}...");
                            continue  # Retry
                    else:
                        self.logger.warning(
                            f"      LLM attempt {attempt + 1} (prompt {idx + 1}): actual_llm_generation is None or not a string. Output: {str(generated_outputs)[:200]}");
                        continue  # Retry

                    question, answer = json_response.get("Question"), json_response.get("Answer")
                    if question is None or not isinstance(question, str) or \
                            answer is None or not isinstance(answer, str):
                        self.logger.warning(
                            f"      LLM response missing Q/A or not strings (prompt {idx + 1}). Parsed: {json_response}");
                        continue  # Retry

                    # Minor cleaning for Q/A
                    question = question.strip(". ") if question.strip().startswith(". ") else question.strip()
                    answer = answer.strip(". ") if answer.strip().startswith(". ") else answer.strip()
                    if not question or not answer:  # Empty Q or A after stripping
                        self.logger.warning(
                            f"      LLM response Q or A empty after stripping (prompt {idx + 1}). Q:'{question}', A:'{answer}'");
                        continue  # Retry

                    # Validate <regionN> tags
                    prompt_tags = set(re.findall(r"<region\d+>", user_prompt_text))
                    question_tags = set(re.findall(r"<region\d+>", question))

                    if prompt_tags == question_tags:  # Exact set match
                        # Check if each tag appears exactly once
                        if all(question.count(tag) == 1 for tag in prompt_tags):
                            q_final, a_final = question, answer;
                            success = True;
                            break  # Success
                        else:
                            self.logger.debug(
                                f"      LLM attempt {attempt + 1} (prompt {idx + 1}): <regionX> count mismatch in Q. Q_tags: {question_tags}, counts: {[question.count(t) for t in prompt_tags]}. Prompt: '{user_prompt_text[:50]}...'")
                    else:
                        self.logger.debug(
                            f"      LLM attempt {attempt + 1} (prompt {idx + 1}): <regionX> set mismatch. Expected: {prompt_tags}, Got in Q: {question_tags}. Prompt: '{user_prompt_text[:50]}...'")

                except Exception as e_inner_llm:
                    llm_api_call_end_exception = time.perf_counter()
                    self.logger.warning(
                        f"      LLM rephrase attempt {attempt + 1} for prompt {idx + 1} ('{user_prompt_text[:50]}...') failed with exception: {e_inner_llm} (took {llm_api_call_end_exception - llm_api_call_start:.4f}s for this attempt)",
                        exc_info=True)

            single_prompt_overall_end_time = time.perf_counter()
            duration_single_prompt_processing = single_prompt_overall_end_time - single_prompt_overall_start_time
            total_llm_processing_time_for_all_prompts += duration_single_prompt_processing

            if success:
                rephrased_conversations.append((q_final, a_final))
                self.logger.info(
                    f"    LLM Rephrased prompt {idx + 1} successfully in {duration_single_prompt_processing:.4f}s. Q: {q_final} || A: {a_final}")
            else:
                self.logger.warning(
                    f"    LLM failed for prompt {idx + 1}: '{user_prompt_text[:100]}...' after {max_retries} attempts. Total time for this prompt: {duration_single_prompt_processing:.4f}s.")

        self.logger.info(
            f"  Finished LLM rephrasing. Total time for {len(rephrased_conversations)} successful rephrases out of {num_prompts_to_process} prompts: {total_llm_processing_time_for_all_prompts:.4f} seconds.")
        return rephrased_conversations

    def __del__(self):
        self.logger.info("Attempting to release resources in GeneralizedSceneGraphGenerator.__del__...")

        if hasattr(self, 'depth_model') and self.depth_model is not None:
            del_depth_start = time.perf_counter()
            try:
                if hasattr(self.depth_model, 'cpu'): self.depth_model.cpu()
                del self.depth_model
                self.depth_model = None
                self.logger.info("  Depth model resources released.")
            except Exception as e:
                self.logger.warning(f"  Error releasing depth model: {e}")
            del_depth_end = time.perf_counter()
            self.logger.info(f"  Depth model cleanup took {del_depth_end - del_depth_start:.4f} seconds.")

        if hasattr(self, 'yoloe_model') and self.yoloe_model is not None:
            del self.yoloe_model  # YOLOE might manage its own CUDA, simple del should be fine
            self.yoloe_model = None
            self.logger.info("  YOLOE model deleted.")

        if hasattr(self, 'llm_pipeline') and self.llm_pipeline is not None:
            del_llm_start = time.perf_counter()
            try:
                if hasattr(self.llm_pipeline, 'model') and self.llm_pipeline.model is not None:
                    if hasattr(self.llm_pipeline.model, 'cpu'): self.llm_pipeline.model.cpu()
                    del self.llm_pipeline.model
                del self.llm_pipeline
                self.llm_pipeline = None;
                self.llm_tokenizer_hf = None
                self.logger.info("  LLM resources released/cleaned.")
            except Exception as e:
                self.logger.warning(f"  Error releasing LLM resources: {e}")
            del_llm_end = time.perf_counter()
            self.logger.info(f"  LLM cleanup took {del_llm_end - del_llm_start:.4f} seconds.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("  torch.cuda.empty_cache() called.")

        self.logger.info("GeneralizedSceneGraphGenerator resource release attempt finished.")


# --- Example Usage (Update config for Qwen2 or Llama3, and Depth Anything V2) ---
if __name__ == "__main__":
    main_script_start_time = time.perf_counter()
    import sys
    import logging

    # Ensure the config file exists and paths within it are correct
    # For Depth Anything V2, 'depth_anything_load_from' is crucial.
    # For YOLOE, 'yoloe_model_path' is crucial.
    # For LLM, 'llm_model_name_hf' (Hugging Face model identifier).

    CONFIG_FILE_NAME = "v2_hf_llm_da2.py"  # Example: Create this config
    # Create a dummy config if it doesn't exist for testing
    if not os.path.exists(CONFIG_FILE_NAME):
        print(f"Warning: Config file {CONFIG_FILE_NAME} not found. Creating a dummy one.")
        # --- Create dummy v2_hf_llm_da2.py ---
        dummy_config_content = f"""
# Dummy Configuration for GeneralizedSceneGraphGenerator with Depth Anything V2
# Paths here are placeholders and MUST be updated by the user.

# General
log_dir = "./temp_outputs_da2/logs"
wis3d_folder = "./temp_outputs_da2/wis3d_visualizations"
vis = True # Enable Wis3D visualization output

# Image Processing
image_resize_height = 512 # Target height for processing images (0 or None to use original)
min_mask_area_pixel = 50  # Minimum pixel area for a detected mask to be considered

# YOLOE Segmentation Model
yoloe_model_path = "yoloe-s-seg-coco.pt" # Placeholder - download a YOLOE model
yoloe_confidence_threshold = 0.25
yoloe_mask_threshold = 0.5
yoloe_mask_dilate_iterations = 0 # e.g., 1 or 2 for slight expansion
yoloe_mask_dilate_kernel_size = 3

# Depth Anything V2 Model
# IMPORTANT: User must download the model and provide the correct path.
# Get models from: https://github.com/LiheYoung/DepthAnythingV2
depth_anything_encoder = 'vits' # 'vits', 'vitb', 'vitl', 'vitg' (smallest for faster testing)
# Example for ViT-S (metric depth):
depth_anything_load_from = './checkpoints/depth_anything_v2_vits.pth' # <<-- USER MUST UPDATE THIS PATH
depth_anything_max_depth = 20.0   # Max depth for metric scaling
depth_model_input_size = 518      # Input size (shorter edge) for DA2's internal processing

# Camera Intrinsics (for depth to point cloud conversion)
# These should correspond to the 'image_resize_height' if resizing is used.
# If None, they will be estimated from FOV.
depth_anything_focal_length_x = None # Example: 525.0 for a 640px width image with ~60deg HFOV
depth_anything_focal_length_y = None # Example: 525.0
camera_hfov_degrees = 70.0 # Horizontal Field of View (used if focal lengths are None)
camera_vfov_degrees = 55.0 # Vertical Field of View

# Point Cloud Processing
pcd_sor_neighbors = 20
pcd_sor_std_ratio = 2.0
pcd_voxel_size = 0.005 # meters
dbscan_remove_noise = True
dbscan_eps = 0.03
dbscan_min_points = 15
min_points_threshold = 30 # Min points for an object before PCD processing
min_points_threshold_after_denoise = 20
bbox_min_volume_threshold = 1e-7 # meters^3
obb_robust = True

# OSDSynth Components (Captioner, Prompters) - paths for LLaVA if used by captioner
# llava_model_path = "liuhaotian/llava-v1.5-7b" # If using LLaVA for captioning
# llava_model_base = None # If LLaVA model is not a delta checkpoint

# LLM for Rephrasing (Hugging Face)
llm_model_name_hf = "Qwen/Qwen2-0.5B-Instruct" # Example: Small LLM for faster testing, or None
# llm_model_name_hf = None # Set to None to disable LLM rephrasing
llm_max_retries = 2
llm_max_new_tokens = 150
llm_temperature = 0.1
llm_top_p = 0.95

# Fact/QA Prompter settings from OSDSynth (if defaults are not suitable)
# e.g. fact_predicates_yaml_path, qa_predicates_yaml_path

# Logging Level (DEBUG, INFO, WARNING, ERROR)
# (Handled by setup_logger, but could be a config)
# logger_level = "INFO"
"""
        # Check for YOLOE model (e.g., yoloe-s-seg-coco.pt)
        yoloe_s_path = "yoloe-s-seg-coco.pt"
        if not os.path.exists(yoloe_s_path):
            print(
                f"Warning: YOLOE model {yoloe_s_path} not found. Please download it (e.g., from Ultralytics assets) for the dummy config to work.")
            # Create an empty file to prevent FileNotFoundError, but YOLOE will fail to load
            open(yoloe_s_path, 'a').close()

        # Check for Depth Anything V2 model (e.g., depth_anything_v2_vits.pth)
        da2_vits_path = './checkpoints/depth_anything_v2_vits.pth'
        os.makedirs("./checkpoints", exist_ok=True)
        if not os.path.exists(da2_vits_path):
            print(f"Warning: Depth Anything V2 model {da2_vits_path} not found. "
                  f"Please download it from https://github.com/LiheYoung/DepthAnythingV2 "
                  f"and place it at the specified path for the dummy config to work.")
            # Create an empty file
            open(da2_vits_path, 'a').close()

        with open(CONFIG_FILE_NAME, "w") as f:
            f.write(dummy_config_content)
        print(f"Created dummy config: {CONFIG_FILE_NAME}. PLEASE REVIEW AND UPDATE PATHS.")

    config_file_path = CONFIG_FILE_NAME  # Use the potentially created dummy config

    demo_image_dir = "./demo_images"
    if not os.path.exists(demo_image_dir): os.makedirs(demo_image_dir)
    demo_image_path = os.path.join(demo_image_dir, "boy_on_chair.jpg")  # A more complex scene

    if not os.path.exists(demo_image_path):
        print(f"Warning: Demo image {demo_image_path} not found. Creating a dummy one.")
        try:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(dummy_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green box
            cv2.rectangle(dummy_img, (300, 200), (450, 350), (255, 0, 0), -1)  # Blue box
            cv2.putText(dummy_img, "Dummy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(demo_image_path, dummy_img)
        except Exception as e:
            print(f"Could not create dummy image: {e}")

    print(f"Using config: {config_file_path}")
    generator_main = None
    try:
        temp_cfg_main = Config.fromfile(config_file_path)
        llm_name_for_init_main = temp_cfg_main.get("llm_model_name_hf")

        print(
            f"Attempting to use LLM: {llm_name_for_init_main if llm_name_for_init_main else 'None (LLM rephrasing disabled)'}")
        print(f"YOLOE model from config: {temp_cfg_main.get('yoloe_model_path')}")
        print(
            f"Depth Anything V2 encoder: {temp_cfg_main.get('depth_anything_encoder')}, model path: {temp_cfg_main.get('depth_anything_load_from')}")
        print(f"Target device for models: cpu (can be changed in script or via config if supported by generator init)")

        generator_init_start_time = time.perf_counter()
        generator_main = GeneralizedSceneGraphGenerator(
            config_path=config_file_path,
            device="cuda" if torch.cuda.is_available() else "cpu",  # Use CUDA if available
            llm_model_name_hf=llm_name_for_init_main,
            llm_device_hf="auto"  # Let transformers pipeline decide LLM device
        )
        generator_init_end_time = time.perf_counter()
        print(
            f"GeneralizedSceneGraphGenerator initialization took {generator_init_end_time - generator_init_start_time:.4f} seconds.")

    except Exception as e_init:
        print(f"FATAL ERROR during generator initialization: {e_init}")
        import traceback;

        traceback.print_exc()
        if generator_main: del generator_main  # Try to clean up
        exit()

    current_image_to_process = demo_image_path

    print(f"\n--- Generating Facts for {current_image_to_process} with YOLOE, DepthAnythingV2 and LLM (if enabled) ---")
    try:
        # Example: custom_vocab = ["person", "chair", "table"] # Or None to detect all
        custom_vocab = None  # Detect all YOLOE classes initially

        overall_facts_gen_start_time = time.perf_counter()
        # Override specific config values for this run if needed:
        # e.g., generator_main.cfg.vis = True
        detected_objects_list, facts, rephrased_qas_f = generator_main.generate_facts(
            current_image_to_process,
            custom_vocabulary=custom_vocab,
            run_llm_rephrase=True if generator_main.llm_pipeline else False,  # Only run if LLM is loaded
            # vis=True # Example of overriding, or set in config file
            # depth_anything_focal_length_x=600 # Example override
        )
        overall_facts_gen_end_time = time.perf_counter()
        print(
            f"Call to generate_facts (from main) took {overall_facts_gen_end_time - overall_facts_gen_start_time:.4f} seconds.")

        if detected_objects_list:
            print(f"\nGenerated {len(detected_objects_list)} DetectedObject instances:")
            for i, obj in enumerate(detected_objects_list):
                print(f"  Object {i + 1}: {obj}")  # Relies on DetectedObject.__repr__

            print(f"\nGenerated {len(facts)} template facts:")
            for i, fact_str in enumerate(facts):
                print(f"  Fact {i + 1}: {fact_str}")

            if rephrased_qas_f:
                print(f"\nGenerated {len(rephrased_qas_f)} LLM-rephrased QAs from facts:")
                for i_qa, (q, a) in enumerate(rephrased_qas_f):
                    print(f"  LLM QA {i_qa + 1}: Q: {q} || A: {a}")
            elif generator_main.llm_pipeline and generator_main.cfg.get("run_llm_rephrase", True):
                print(
                    "LLM rephrasing was enabled (or intended), but no QAs were generated from facts (either no facts, no suitable prompts, or LLM failed).")
        else:
            print("No DetectedObject instances generated (likely no valid detections or processing failed).")

    except SkipImageException as e_skip:
        print(f"Skipped image processing for {current_image_to_process}: {e_skip}")
    except Exception as e_facts:
        print(f"Error in fact generation example for {current_image_to_process}: {e_facts}")
        import traceback;

        traceback.print_exc()

    # --- Example for QA Generation ---
    print(f"\n--- Generating QAs for {current_image_to_process} (No LLM rephrasing in this part) ---")
    try:
        custom_vocab_qa = ["person", "dog", "car"]  # Different vocab for QA example
        overall_qa_gen_start_time = time.perf_counter()
        detected_objects_list_qa, template_qas_direct = generator_main.generate_qa(
            current_image_to_process,
            custom_vocabulary=custom_vocab_qa,  # Can be None
            # vis=False # Example: turn off vis for this specific call
        )
        overall_qa_gen_end_time = time.perf_counter()
        print(
            f"Call to generate_qa (from main) took {overall_qa_gen_end_time - overall_qa_gen_start_time:.4f} seconds.")

        if detected_objects_list_qa:
            print(f"\n(QA Flow) Generated {len(detected_objects_list_qa)} DetectedObject instances:")
            # for i, obj in enumerate(detected_objects_list_qa):
            #     print(f"  (QA) Object {i+1}: {obj.class_name}, {obj.description[:30]}...")

            if template_qas_direct:
                print(f"\nGenerated {len(template_qas_direct)} direct template QAs:")
                for i_qa_d, (q_d, a_d) in enumerate(template_qas_direct):
                    print(f"  Template QA {i_qa_d + 1}: Q: {q_d} || A: {a_d}")
            else:
                print("No direct template QAs generated.")
        else:
            print("(QA Flow) No DetectedObject instances generated.")

    except SkipImageException as e_skip_qa:
        print(f"Skipped QA generation for {current_image_to_process}: {e_skip_qa}")
    except Exception as e_qa_gen:
        print(f"Error in QA generation example for {current_image_to_process}: {e_qa_gen}")
        import traceback;

        traceback.print_exc()

    del_gen_start_time = time.perf_counter()
    if generator_main:
        del generator_main
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection
    del_gen_end_time = time.perf_counter()
    print(f"\nGenerator cleanup took {del_gen_end_time - del_gen_start_time:.4f} seconds.")

    main_script_end_time = time.perf_counter()
    print(
        f"Processing complete. Total script execution time: {main_script_end_time - main_script_start_time:.4f} seconds.")