import os
import cv2
import numpy as np
from PIL import Image
import torch
# import torchvision # Not directly used in the critical path after YOLOE change
import time
import json
import re
import warnings
from mmengine import Config
import gc
# import os # Already imported
# import shutil # Not directly used in critical path
# import time # Already imported
# from datetime import datetime # Not in critical path
# from math import pi # Not in critical path

# import gradio as gr # Not used in this script execution
# import numpy as np # Already imported
# import torch # Already imported
# import trimesh # Not directly used in critical path

from unik3d.models import UniK3D
# from unik3d.utils.camera import OPENCV, Fisheye624, Pinhole, Spherical # Not directly used
import open3d as o3d

# import open3d as o3d # Should already be there
from wis3d import Wis3D
import matplotlib  # For color_by_instance (only if vis=True)
from scipy.spatial.transform import Rotation  # For oriented_bbox_to_center_euler_extent (only if vis=True)
from collections import Counter  # For pcd_denoise_dbscan

# YOLOE CHANGE: Import YOLOE
from ultralytics import YOLOE

# OSDSUTILS imports (ensure osdsynth is in PYTHONPATH or installed)
try:
    from osdsynth.processor.captions import CaptionImage
    # from osdsynth.processor.pointcloud import PointCloudReconstruction # Not used
    from osdsynth.processor.prompt import PromptGenerator as QAPromptGenerator
    from osdsynth.processor.instruction import PromptGenerator as FactPromptGenerator
    from osdsynth.utils.logger import SkipImageException, setup_logger

    OSDSYNTH_AVAILABLE = True
except ImportError as e:
    warnings.warn(
        f"Failed to import osdsynth components: {e}. Ensure osdsynth is correctly installed and in PYTHONPATH.")
    OSDSYNTH_AVAILABLE = False


    class CaptionImage:  # Dummy
        def __init__(self, *args, **kwargs): pass

        def process_local_caption(self, detection_list_dicts):
            # Simple fallback: use class_name as caption
            for det in detection_list_dicts:
                det["caption"] = det.get("class_name", "Unknown Object")
            return detection_list_dicts


    class QAPromptGenerator:  # Dummy
        def __init__(self, *args, **kwargs): pass

        def evaluate_predicates_on_pairs(self, detection_list_dicts): return []


    class FactPromptGenerator:  # Dummy
        def __init__(self, *args, **kwargs): pass

        def evaluate_predicates_on_pairs(self, detection_list_dicts): return []


    class SkipImageException(Exception):
        pass


    def setup_logger(name="dummy_sg_gen"):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

# Hugging Face Transformers imports (LLM likely disabled for RPi, but keep for completeness)
try:
    from transformers import pipeline, AutoTokenizer  # , AutoModelForCausalLM

    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    warnings.warn("Hugging Face Transformers not found. LLM rephrasing will not be available.")


# --- Helper: Cropping (can be kept minimal if osdsynth missing) ---
def crop_detections_with_xyxy_local(cfg, image_pil, detections_list):
    """
    A local, simplified version of crop_detections_with_xyxy if osdsynth is not available or
    if a more lightweight version is preferred.
    This version will not perform mask-based modifications like blackout/red_outline
    as those add processing time. It just crops.
    """
    for idx, detection in enumerate(detections_list):
        x1, y1, x2, y2 = detection["xyxy"]

        # Ensure PIL image for cropping
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(image_pil)

            # Round coordinates and ensure they are integers
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

        # Define the cropping coordinates with padding
        padding = cfg.get("crop_padding", 10)  # Add a config for padding if needed
        img_w, img_h = image_pil.size

        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(img_w, x2 + padding)
        crop_y2 = min(img_h, y2 + padding)

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            image_crop_pil = None  # Invalid crop box
        else:
            image_crop_pil = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        detections_list[idx]["image_crop"] = image_crop_pil
        # mask_crop and image_crop_modified are removed for simplicity here.
        # If needed, they can be added back but will increase processing.
    return detections_list


# --- Definition of the DetectedObject class ---
class DetectedObject:
    def __init__(self,
                 class_name: str,
                 description: str,
                 segmentation_mask_2d: np.ndarray,  # Kept, but might be large
                 bounding_box_2d: np.ndarray,
                 point_cloud_3d: o3d.geometry.PointCloud = None,  # Optional for speed
                 bounding_box_3d_oriented: o3d.geometry.OrientedBoundingBox = None,  # Optional
                 bounding_box_3d_axis_aligned: o3d.geometry.AxisAlignedBoundingBox = None,  # Optional
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
        num_points = 0
        if self.point_cloud_3d and self.point_cloud_3d.has_points():
            num_points = len(self.point_cloud_3d.points)

        obb_center_str = 'N/A'
        if self.bounding_box_3d_oriented and not self.bounding_box_3d_oriented.is_empty():
            obb_center_str = f"[{self.bounding_box_3d_oriented.center[0]:.2f}, ...]"

        return (f"<DetectedObject: {self.class_name} "
                f"(Desc: '{self.description[:20]}...'), "
                f"2D_bbox: [{int(self.bounding_box_2d[0])}...], "
                f"Mask_Shape: {self.segmentation_mask_2d.shape if self.segmentation_mask_2d is not None else 'N/A'}, "
                f"3D_pts: {num_points}, "
                f"3D_OBB_center: {obb_center_str}>")


# --- Helper functions for Point Cloud Processing and Visualization (Optimized for RPi) ---
def process_pcd_for_unik3d(cfg, pcd, run_dbscan=True):
    if not pcd.has_points() or len(pcd.points) == 0: return pcd

    # Statistical Outlier Removal (SOR) - can be costly
    if cfg.get("pcd_enable_sor", False):  # Made configurable, default False for RPi
        try:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg.get("pcd_sor_neighbors", 10),
                std_ratio=cfg.get("pcd_sor_std_ratio", 2.0)
            )
        except RuntimeError:
            pass  # Ignore if it fails
        if not pcd.has_points() or len(pcd.points) == 0: return pcd

    # Voxel Downsampling - good for performance
    voxel_size = cfg.get("pcd_voxel_size", 0.05)  # Increased default for RPi
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if not pcd.has_points() or len(pcd.points) == 0: return pcd

    # DBSCAN - very costly, recommend disabling for RPi
    if cfg.get("dbscan_remove_noise", False) and run_dbscan:  # Default False for RPi
        pcd = pcd_denoise_dbscan_for_unik3d(
            pcd,
            eps=cfg.get("dbscan_eps", 0.1),
            min_points=cfg.get("dbscan_min_points", 5)
        )
    return pcd


def pcd_denoise_dbscan_for_unik3d(pcd: o3d.geometry.PointCloud, eps=0.1, min_points=5) -> o3d.geometry.PointCloud:
    if not pcd.has_points() or len(pcd.points) < min_points: return pcd
    try:
        # print_progress=False is important for speed
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    except RuntimeError:
        return pcd

    if len(labels) == 0: return o3d.geometry.PointCloud()  # No labels produced

    counts = Counter(labels)
    if -1 in counts: del counts[-1]  # Remove noise label
    if not counts: return o3d.geometry.PointCloud()  # No clusters found

    largest_cluster_label = counts.most_common(1)[0][0]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    if len(largest_cluster_indices) < min_points: return o3d.geometry.PointCloud()
    return pcd.select_by_index(largest_cluster_indices)


def get_bounding_box_for_unik3d(cfg, pcd):
    if not pcd.has_points() or len(pcd.points) < 3:  # Need at least 3 points for a robust bbox
        aabb = o3d.geometry.AxisAlignedBoundingBox()
        obb = o3d.geometry.OrientedBoundingBox()
        return aabb, obb

    axis_aligned_bbox = pcd.get_axis_aligned_bounding_box()

    # Robust OBB can be slow. For RPi, non-robust might be acceptable.
    try:
        oriented_bbox = pcd.get_oriented_bounding_box(robust=cfg.get("obb_robust", False))  # Default False for RPi
    except RuntimeError:  # Fallback if robust fails or for very few points
        oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
    return axis_aligned_bbox, oriented_bbox


# Visualization helpers (only used if vis=True, so less critical for RPi speed if vis=False)
def color_by_instance_for_unik3d(pcds):
    if not pcds: return []
    cmap = matplotlib.colormaps.get_cmap("turbo")
    instance_colors = cmap(np.linspace(0, 1, len(pcds)))
    colored_pcds = []
    for i, pcd_original in enumerate(pcds):
        if pcd_original.has_points():
            pcd_copy = o3d.geometry.PointCloud(pcd_original)
            pcd_copy.colors = o3d.utility.Vector3dVector(np.tile(instance_colors[i, :3], (len(pcd_copy.points), 1)))
            colored_pcds.append(pcd_copy)
        else:
            colored_pcds.append(o3d.geometry.PointCloud())
    return colored_pcds


def oriented_bbox_to_center_euler_extent_for_unik3d(bbox_center, box_R, bbox_extent):
    center = np.asarray(bbox_center)
    extent = np.asarray(bbox_extent)
    eulers = Rotation.from_matrix(box_R.copy()).as_euler("XYZ")
    return center, eulers, extent


def axis_aligned_bbox_to_center_euler_extent_for_unik3d(min_coords, max_coords):
    center = tuple((min_val + max_val) / 2.0 for min_val, max_val in zip(min_coords, max_coords))
    eulers = (0.0, 0.0, 0.0)
    extent = tuple(abs(max_val - min_val) for min_val, max_val in zip(min_coords, max_coords))
    return center, eulers, extent


# --- End of Helper functions ---

warnings.filterwarnings("ignore")


def instantiate_model(model_name):
    type_ = model_name[0].lower()
    name = f"unik3d-vit{type_}"
    # For RPi, ensure model files are downloaded and path is correct if not using from_pretrained directly
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    model.resolution_level = 9  # This is default; lower might be faster but less accurate if model supports it.
    model.interpolation_mode = "bilinear"  # Could try "nearest" but "bilinear" is common.
    device = torch.device("cpu")  # FORCE CPU
    model = model.to(device).eval()
    return model


def prepare_llm_prompts_from_facts(facts, detection_list_dicts):  # Unlikely to be used on RPi
    # ... (original implementation, as LLM is usually off) ...
    # This function is lightweight, so no major optimization needed here.
    # The bottleneck is running the LLM, not preparing prompts.
    batched_instructions = []
    # ... (rest of function as in original)
    return batched_instructions


def parse_qas_from_vqa_results(vqa_results):  # Lightweight
    conversations = []
    for item in vqa_results:
        qa_pair = item[0]
        conversations.append(qa_pair)
    return conversations


LLM_HF_SYSTEM_PROMPT = r"""You are a helpful AI assistant. Your task is to rephrase a given factual description about objects in an image into a natural question and answer pair.
The input description will use tags like <region0>, <region1> to refer to specific objects.
Your output MUST be a single JSON object containing two keys: "Question" and "Answer".
The "Question" should incorporate all the <regionX> tags from the input. Each tag must appear exactly once in the question.
The "Answer" should naturally respond to the question, also using the <regionX> tags if appropriate, or referring to the objects they represent.
Example Input:
[Objects]: <region0> person, <region1> car. [Description]: <region0> is standing next to <region1>.
Example Output:
```json
{
  "Question": "What is <region0> (person) doing in relation to <region1> (car)?",
  "Answer": "<region0> (person) is standing next to <region1> (car)."
}
```
Ensure the output is ONLY the JSON object, starting with ```json and ending with ```."""


class GeneralizedSceneGraphGenerator:

    def __init__(self, config_path="config/v2_hf_qwen.py", device="cpu",
                 llm_model_name_hf=None, llm_device_hf="cpu"):

        self.logger = setup_logger(name="SGG_RPi")  # Use the RPi-specific logger
        self.logger.info("Initializing GeneralizedSceneGraphGenerator for RPi (CPU forced)...")
        _start_time_init = time.time()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.cfg = Config.fromfile(config_path)
        self.device = torch.device("cpu")  # Force CPU
        self.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        # YOLOE Initialization
        _start_yoloe_init = time.time()
        self.yoloe_model_path = self.cfg.get("yoloe_model_path", "yoloe-s-seg.pt")  # Default to a smaller one
        try:
            self.logger.info(f"Initializing YOLOE model from: {self.yoloe_model_path}")
            self.yoloe_model = YOLOE(self.yoloe_model_path)
            # YOLOE model's device setting is usually handled internally or via .to(device) if it's a torch.nn.Module.
            # The `predict` call will use the device it's on. We ensure overall script context is CPU.
            # If YOLOE allows, self.yoloe_model.to(self.device) would be explicit.
            # For ultralytics, it typically auto-selects or can be passed in predict. Forcing CPU here.
            if hasattr(self.yoloe_model, 'model') and hasattr(self.yoloe_model.model, 'to'):
                self.yoloe_model.model.to(self.device)
            elif hasattr(self.yoloe_model, 'to'):
                self.yoloe_model.to(self.device)

            self.logger.info(f"YOLOE model initialized. Time: {time.time() - _start_yoloe_init:.2f}s")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOE model: {e}", exc_info=True)
            raise RuntimeError("Could not initialize YOLOE model.")

        # UniK3D Initialization
        _start_unik3d_init = time.time()
        if self.cfg.get("enable_unik3d", True):  # Configurable enable/disable
            try:
                self.logger.info("Initializing UniK3D model (Small)...")
                self.unik3d_model = instantiate_model("Small")  # Already forced to CPU in instantiate_model
                self.logger.info(f"UniK3D model initialized. Time: {time.time() - _start_unik3d_init:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to initialize UniK3D model: {e}", exc_info=True)
                self.unik3d_model = None  # Allow running without it if init fails but enable_unik3d was true
                self.logger.warning(
                    "UniK3D model initialization failed. Proceeding without 3D capabilities if possible.")
        else:
            self.unik3d_model = None
            self.logger.info("UniK3D model is disabled by config.")

        # OSDSynth Components (Captioner, Prompters)
        _start_osds_init = time.time()
        if OSDSYNTH_AVAILABLE:
            self.captioner = CaptionImage(self.cfg, self.logger, self.device,
                                          init_lava=False)  # LLaVA is heavy, ensure False
            self.qa_prompter = QAPromptGenerator(self.cfg, self.logger, self.device)
            self.fact_prompter = FactPromptGenerator(self.cfg, self.logger, self.device)
        else:  # Use dummy versions if osdsynth not available
            self.captioner = CaptionImage()
            self.qa_prompter = QAPromptGenerator()
            self.fact_prompter = FactPromptGenerator()
        self.logger.info(
            f"OSDSynth components (Captioner, Prompters) initialized. Time: {time.time() - _start_osds_init:.2f}s")

        # LLM Initialization (Likely disabled for RPi)
        self.llm_pipeline = None
        self.llm_tokenizer_hf = None
        llm_model_name_hf_resolved = self.cfg.get("llm_model_name_hf", llm_model_name_hf)  # Prioritize config
        if llm_model_name_hf_resolved and HF_TRANSFORMERS_AVAILABLE and self.cfg.get("enable_llm", False):
            _start_llm_init = time.time()
            self.logger.info(f"Initializing Hugging Face LLM: {llm_model_name_hf_resolved} on CPU...")
            try:
                self.llm_tokenizer_hf = AutoTokenizer.from_pretrained(llm_model_name_hf_resolved,
                                                                      trust_remote_code=True)
                if self.llm_tokenizer_hf.pad_token is None:
                    self.llm_tokenizer_hf.pad_token = self.llm_tokenizer_hf.eos_token or self.llm_tokenizer_hf.unk_token

                self.llm_pipeline = pipeline("text-generation", model=llm_model_name_hf_resolved,
                                             tokenizer=self.llm_tokenizer_hf,
                                             device_map="cpu",  # Explicitly CPU
                                             torch_dtype=torch.float32,
                                             trust_remote_code=True)
                if hasattr(self.llm_pipeline, 'model'): self.llm_pipeline.model.to("cpu")  # Ensure model on CPU
                self.logger.info(
                    f"HF LLM pipeline for {llm_model_name_hf_resolved} initialized. Time: {time.time() - _start_llm_init:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to initialize HF LLM pipeline: {e}", exc_info=True)
                self.llm_pipeline = None;
                self.llm_tokenizer_hf = None
        elif llm_model_name_hf_resolved and self.cfg.get("enable_llm", False) and not HF_TRANSFORMERS_AVAILABLE:
            self.logger.warning("LLM configured but Transformers lib not found.")
        else:
            self.logger.info("LLM rephrasing is disabled by config or missing model name.")

        default_wis3d_folder = os.path.join(self.cfg.get("log_dir", "./temp_outputs_rpi/log"),
                                            f"Wis3D_RPi_{self.timestamp}")
        self.cfg.wis3d_folder = self.cfg.get("wis3d_folder", default_wis3d_folder)
        if self.cfg.get("vis", False): os.makedirs(self.cfg.wis3d_folder, exist_ok=True)

        self.logger.info(f"Total SGG Initialization Time: {time.time() - _start_time_init:.2f}s")

    def _override_config_and_reinit(self, **kwargs):
        # This function might be less used in a fixed RPi setup, but keep for flexibility
        # Re-initialization can be slow, so avoid frequent calls.
        # For RPi, it's better to set optimal config once.
        # Simplified: only re-init captioner if relevant keys change
        reinit_captioner = False
        for key, value in kwargs.items():
            parts = key.split('.');
            cfg_node = self.cfg;
            changed = False
            try:
                for i, part in enumerate(parts[:-1]): cfg_node = cfg_node[part]
                if cfg_node.get(parts[-1]) != value: cfg_node[parts[-1]] = value; changed = True
            except KeyError:
                cfg_node[parts[-1]] = value; changed = True  # Add new key

            if changed:
                if key.startswith("llava_") or key.startswith("global_qs_list") or key.startswith(
                        "caption_"):  # Captioner related
                    reinit_captioner = True
                # YOLOE model path change handling (if you want to support dynamic YOLOE model change)
                # if key == "yoloe_model_path":
                #     self.logger.info("YOLOE model path changed, re-initializing YOLOE...")
                #     self.yoloe_model_path = value
                #     self.yoloe_model = YOLOE(self.yoloe_model_path)
                #     if hasattr(self.yoloe_model, 'model') and hasattr(self.yoloe_model.model, 'to'):
                #         self.yoloe_model.model.to(self.device)

        if reinit_captioner and OSDSYNTH_AVAILABLE:
            self.logger.info("Re-initializing CaptionImage due to config change.")
            init_lava_flag = hasattr(self.captioner, 'llava_processor') and self.captioner.llava_processor is not None
            self.captioner = CaptionImage(self.cfg, self.logger, self.device, init_lava=init_lava_flag)

    def _load_image(self, image_input):
        _start_time = time.time()
        if isinstance(image_input, str):
            if not os.path.exists(image_input): raise FileNotFoundError(f"Image not found at {image_input}")
            image_bgr = cv2.imread(image_input)
            if image_bgr is None: raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, np.ndarray):
            image_bgr = image_input.copy()
        else:
            raise TypeError("image_input must be a file path (str) or a NumPy array (BGR).")

        h, w = image_bgr.shape[:2]
        if h == 0 or w == 0: raise ValueError("Image has zero height or width.")

        target_h = self.cfg.get("image_resize_height", 320)  # RPi default
        if target_h > 0 and h != target_h:  # Only resize if necessary and target_h is set
            scale = target_h / h
            target_w = int(w * scale)
            image_bgr_resized = cv2.resize(image_bgr, (target_w, target_h),
                                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        else:
            image_bgr_resized = image_bgr  # No resize if target_h is 0, or image is already target_h

        self.logger.info(
            f"Image loaded and resized to {image_bgr_resized.shape[:2]}. Time: {time.time() - _start_time:.3f}s")
        return image_bgr_resized

    def _get_object_classes(self, image_rgb_pil, custom_vocabulary=None):  # image_rgb_pil not used by YOLOE for this
        # YOLOE classes are fixed by the model.
        yoloe_class_names = list(self.yoloe_model.names.values())

        if custom_vocabulary:
            # Validate custom_vocabulary (must be list of strings)
            if not isinstance(custom_vocabulary, list) or not all(isinstance(s, str) for s in custom_vocabulary):
                self.logger.warning("Invalid custom_vocabulary format. Using all YOLOE classes.")
                return yoloe_class_names
            if not custom_vocabulary:  # Empty list
                self.logger.warning("Empty custom_vocabulary list. Using all YOLOE classes.")
                return yoloe_class_names

            valid_custom_classes = [cls for cls in custom_vocabulary if cls in yoloe_class_names]
            if not valid_custom_classes:
                self.logger.warning(f"None of custom vocabulary {custom_vocabulary} in YOLOE classes. Using all.")
                return yoloe_class_names
            self.logger.info(f"Using custom vocabulary (YOLOE filtered): {valid_custom_classes}")
            return valid_custom_classes
        else:
            self.logger.info(f"Using all YOLOE detectable classes: {len(yoloe_class_names)} classes.")
            return yoloe_class_names

    def _segment_image(self, image_bgr, classes_to_detect):
        _start_time = time.time()
        # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # YOLOE predict takes BGR
        image_rgb_pil_for_crop = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        target_class_indices = None
        if classes_to_detect:
            name_to_idx_map = {name: idx for idx, name in self.yoloe_model.names.items()}
            target_class_indices = [name_to_idx_map[name] for name in classes_to_detect if name in name_to_idx_map]
            if not target_class_indices:
                self.logger.warning(f"No valid YOLOE class indices from {classes_to_detect}. Detecting all.")

        yoloe_input_size = self.cfg.get("yoloe_input_size", image_bgr.shape[0])
        yoloe_conf = self.cfg.get("yoloe_confidence_threshold", 0.4)  # RPi tuned

        # Forcing YOLOE to use CPU during predict call if possible with ultralytics
        predict_device = "cpu"  # self.device

        _t_yolo_predict_start = time.time()
        yolo_results = self.yoloe_model.predict(
            source=image_bgr,  # YOLOE ultralytics usually handles BGR/RGB internally
            classes=target_class_indices,
            conf=yoloe_conf,
            imgsz=yoloe_input_size,
            device=predict_device,  # Explicitly tell predict to use CPU
            verbose=False  # Reduce YOLOE console output
        )
        _t_yolo_predict_end = time.time()
        self.logger.info(f"YOLOE model.predict call. Time: {_t_yolo_predict_end - _t_yolo_predict_start:.3f}s")

        if not yolo_results or not yolo_results[0].boxes or yolo_results[0].boxes.shape[0] == 0:
            raise SkipImageException(f"No objects detected by YOLOE (conf={yoloe_conf}, imgsz={yoloe_input_size}).")

        res = yolo_results[0]
        if res.masks is None or res.masks.data is None or res.masks.data.shape[0] == 0:
            raise SkipImageException(f"No masks in YOLOE results.")

        _t_mask_proc_start = time.time()
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # BBoxes
        confidences = res.boxes.conf.cpu().numpy()  # Confidences
        class_ids = res.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Use res.masks.masks if available (already resized binary masks)
        # Otherwise, process res.masks.data (raw feature map masks, need upscaling)
        if hasattr(res.masks, 'masks') and res.masks.masks is not None:
            # masks are [N, H, W] tensor of booleans or 0/1 on CPU
            masks_np_binary = res.masks.masks.cpu().numpy().astype(bool)
        else:  # Fallback to processing masks.data
            masks_data_np = res.masks.data.cpu().numpy()  # [N, H_feat, W_feat] or [N, H_img, W_img]
            h_img, w_img = image_bgr.shape[:2]
            processed_masks_list = []
            for i in range(masks_data_np.shape[0]):
                mask_i = masks_data_np[i, :, :]
                if mask_i.shape[0] != h_img or mask_i.shape[1] != w_img:
                    # This resize is slow, try to use YOLOE models that output full-res masks or use .masks.masks
                    mask_i_resized = cv2.resize(mask_i, (w_img, h_img),
                                                interpolation=cv2.INTER_NEAREST)  # Nearest for masks
                else:
                    mask_i_resized = mask_i
                binary_mask = (mask_i_resized > self.cfg.get("yoloe_mask_threshold", 0.5)).astype(bool)
                processed_masks_list.append(binary_mask)
            if not processed_masks_list: raise SkipImageException("Mask processing failed (data to binary).")
            masks_np_binary = np.stack(processed_masks_list)

        _t_mask_proc_end = time.time()
        self.logger.debug(
            f"YOLOE mask data extraction & binarization. Time: {_t_mask_proc_end - _t_mask_proc_start:.3f}s")

        detected_class_names = [self.yoloe_model.names[cid] for cid in class_ids]
        detection_list = []
        for i in range(len(boxes_xyxy)):
            detection_list.append({
                "xyxy": boxes_xyxy[i],
                "mask": masks_np_binary[i],
                "subtracted_mask": masks_np_binary[i].copy(),  # For now, same as mask
                "confidence": confidences[i],
                "class_name": detected_class_names[i],
                "class_id": class_ids[i]
            })

        if not detection_list: raise SkipImageException("No detections formulated into list.")

        _t_filter_sort_start = time.time()
        # Filter by mask area (fast operation)
        filtered_detection_list = []
        min_area = self.cfg.get("min_mask_area_pixel", 200)  # RPi tuned
        for det in detection_list:
            mask_area = np.sum(det["mask"])  # np.sum on bool array is fast
            if mask_area >= min_area:
                filtered_detection_list.append(det)

        if not filtered_detection_list: raise SkipImageException("No detections after area filtering.")
        detection_list = filtered_detection_list

        # Sort by area (can be skipped if order doesn't matter and many objects)
        if self.cfg.get("sort_detections_by_area", True):
            detection_list = sorted(detection_list, key=lambda d: np.sum(d["mask"]), reverse=True)

        # Mask dilation (costly, disable for RPi by default)
        if self.cfg.get("yoloe_mask_dilate_iterations", 0) > 0:
            _t_dilate_start = time.time()
            kernel_size = self.cfg.get("yoloe_mask_dilate_kernel_size", 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            iterations = self.cfg.get("yoloe_mask_dilate_iterations")
            for det in detection_list:
                mask_uint8 = det["mask"].astype(np.uint8) * 255  # OpenCV needs 0 or 255
                dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=iterations)
                det["mask"] = (dilated_mask > 0).astype(bool)
                det["subtracted_mask"] = det["mask"].copy()
            self.logger.debug(f"Mask dilation. Time: {time.time() - _t_dilate_start:.3f}s")

        _t_filter_sort_end = time.time()
        self.logger.debug(f"Detection filtering & sorting. Time: {_t_filter_sort_end - _t_filter_sort_start:.3f}s")

        # Crop detections (relatively fast if not modifying pixels)
        _t_crop_start = time.time()
        # Use local simple cropper for RPi if osdsynth not available or for max speed
        detection_list = crop_detections_with_xyxy_local(self.cfg, image_rgb_pil_for_crop, detection_list)
        self.logger.debug(f"Cropping detections. Time: {time.time() - _t_crop_start:.3f}s")

        self.logger.info(
            f"Segmentation (YOLOE) successful: {len(detection_list)} objects. Total Time: {time.time() - _start_time:.3f}s")
        return detection_list

    def _process_common(self, image_input, custom_vocabulary=None, **kwargs):
        _start_time_common = time.time()
        self.logger.info("Starting common processing pipeline...")

        # Config override (usually not needed frequently for RPi)
        if kwargs: self._override_config_and_reinit(**kwargs)

        # 1. Load Image
        image_bgr = self._load_image(image_input)  # Resized BGR
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # For UniK3D & PIL uses

        filename_prefix = "rpi_proc_" + self.timestamp
        if isinstance(image_input, str):
            filename_prefix = os.path.splitext(os.path.basename(image_input))[0] + "_rpi_" + self.timestamp

        # 2. Get Object Classes (fast, dictionary lookup)
        _t_get_cls_start = time.time()
        object_classes_to_detect = self._get_object_classes(None, custom_vocabulary)  # PIL image not used
        self.logger.debug(f"Class determination. Time: {time.time() - _t_get_cls_start:.4f}s")

        # 3. Segment Image (YOLOE - this is a major step, already timed internally)
        detection_list_initial_dicts = self._segment_image(image_bgr, object_classes_to_detect)

        if not detection_list_initial_dicts:
            raise SkipImageException("Segmentation (YOLOE) resulted in no initial detections.")

        # --- The rest of _process_common remains largely the same,
        # as it operates on the detection_list_initial_dicts structure ---

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.unik3d_model.infer(image_tensor, camera=None, normalize=True)
        points_3d_global = outputs["points"].squeeze().permute(1, 2, 0).cpu().numpy()

        wis3d_instance = None
        if self.cfg.get("vis", False):
            wis3d_instance = Wis3D(self.cfg.wis3d_folder, filename_prefix)
            if points_3d_global.shape[:2] == image_rgb.shape[:2]:
                wis3d_instance.add_point_cloud(
                    vertices=points_3d_global.reshape((-1, 3)),
                    colors=image_rgb.reshape(-1, 3),
                    name="unik3d_global_scene_pts")
            else:
                self.logger.warning("Global point cloud and image dimensions mismatch for Wis3D coloring.")

        valid_detections_dicts = []
        min_initial_pts = self.cfg.get("min_points_threshold", 20)
        min_processed_pts = self.cfg.get("min_points_threshold_after_denoise", 10)
        min_bbox_volume = self.cfg.get("bbox_min_volume_threshold", 1e-6)

        for det_idx, det_dict in enumerate(detection_list_initial_dicts):  # det_dict is from YOLOE processing
            # `subtracted_mask` is now directly from YOLOE (or post-processed copy of YOLOE mask)
            object_mask = det_dict["subtracted_mask"]
            if not np.any(object_mask):
                self.logger.debug(f"Skipping det {det_idx} ({det_dict.get('class_name', 'N/A')}) due to empty mask.")
                continue

            obj_points_from_global = points_3d_global[object_mask]
            obj_colors_from_global = image_rgb[object_mask] / 255.0

            if len(obj_points_from_global) < min_initial_pts:
                self.logger.debug(f"Skipping det {det_idx} ({det_dict.get('class_name', 'N/A')}): "
                                  f"Too few initial points ({len(obj_points_from_global)} < {min_initial_pts}).")
                continue

            obj_points_from_global += np.random.normal(0, self.cfg.get("pcd_perturb_std", 1e-3),
                                                       obj_points_from_global.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points_from_global)
            pcd.colors = o3d.utility.Vector3dVector(obj_colors_from_global)
            processed_pcd = process_pcd_for_unik3d(self.cfg, pcd)

            if not processed_pcd.has_points() or len(processed_pcd.points) < min_processed_pts:
                self.logger.debug(f"Skipping det {det_idx} ({det_dict.get('class_name', 'N/A')}): "
                                  f"Too few points after processing ({len(processed_pcd.points)} < {min_processed_pts}).")
                continue

            axis_aligned_bbox, oriented_bbox = get_bounding_box_for_unik3d(self.cfg, processed_pcd)
            if axis_aligned_bbox.is_empty() or axis_aligned_bbox.volume() < min_bbox_volume:
                self.logger.debug(f"Skipping det {det_idx} ({det_dict.get('class_name', 'N/A')}): "
                                  f"BBox volume too small ({axis_aligned_bbox.volume()}).")
                continue

            det_dict["pcd"] = processed_pcd
            det_dict["axis_aligned_bbox"] = axis_aligned_bbox
            det_dict["oriented_bbox"] = oriented_bbox
            valid_detections_dicts.append(det_dict)  # This list of dicts is passed to captioner

        if not valid_detections_dicts:
            raise SkipImageException("No valid 3D objects found after processing.")

        if wis3d_instance:
            object_pcds_for_vis = [d["pcd"] for d in valid_detections_dicts]
            instance_colored_pcds = color_by_instance_for_unik3d(object_pcds_for_vis)
            for i, det_data_dict in enumerate(valid_detections_dicts):
                obj_pcd_colored = instance_colored_pcds[i]
                class_name = det_data_dict.get("class_name", f"object_{i}")
                if obj_pcd_colored.has_points():
                    wis3d_instance.add_point_cloud(vertices=np.asarray(obj_pcd_colored.points),
                                                   colors=np.asarray(obj_pcd_colored.colors),
                                                   name=f"{i:02d}_{class_name}_unik3d_pts")
                aa_bbox = det_data_dict["axis_aligned_bbox"]
                if not aa_bbox.is_empty():
                    aa_center, aa_eulers, aa_extent = axis_aligned_bbox_to_center_euler_extent_for_unik3d(
                        aa_bbox.get_min_bound(), aa_bbox.get_max_bound())
                    wis3d_instance.add_boxes(positions=np.array([aa_center]), eulers=np.array([aa_eulers]),
                                             extents=np.array([aa_extent]), name=f"{i:02d}_{class_name}_unik3d_aa_bbox")
                or_bbox = det_data_dict["oriented_bbox"]
                if not or_bbox.is_empty() and np.all(np.array(or_bbox.extent) > 1e-6):
                    or_center, or_eulers, or_extent = oriented_bbox_to_center_euler_extent_for_unik3d(
                        or_bbox.center, or_bbox.R, or_bbox.extent)
                    wis3d_instance.add_boxes(positions=np.array([or_center]), eulers=np.array([or_eulers]),
                                             extents=np.array([or_extent]), name=f"{i:02d}_{class_name}_unik3d_or_bbox")

        captioned_detections_dicts = self.captioner.process_local_caption(valid_detections_dicts)

        detected_object_instances = []
        for det_dict in captioned_detections_dicts:
            description = det_dict.get("caption", det_dict.get("class_name", "Unknown Object"))
            pcd_o3d = det_dict.get("pcd");
            obb_o3d = det_dict.get("oriented_bbox");
            aabb_o3d = det_dict.get("axis_aligned_bbox")
            mask_2d_np = det_dict.get("subtracted_mask", det_dict.get("mask"))
            bbox_2d_np = det_dict.get("xyxy")

            if not all([pcd_o3d, obb_o3d, aabb_o3d, mask_2d_np is not None, bbox_2d_np is not None]):
                self.logger.warning(
                    f"Skipping object '{det_dict.get('class_name', 'N/A')}' due to missing critical data for DetectedObject.")
                continue

            obj_instance = DetectedObject(class_name=det_dict.get("class_name", "Unknown"), description=description,
                                          segmentation_mask_2d=mask_2d_np, bounding_box_2d=bbox_2d_np,
                                          point_cloud_3d=pcd_o3d, bounding_box_3d_oriented=obb_o3d,
                                          bounding_box_3d_axis_aligned=aabb_o3d,
                                          image_crop_pil=det_dict.get("image_crop"))
            detected_object_instances.append(obj_instance)

        if not detected_object_instances:
            raise SkipImageException("No valid DetectedObject instances could be created.")

        return detected_object_instances, captioned_detections_dicts, filename_prefix

    # generate_facts (NO CHANGE in its internal logic other than what _process_common implies)
    def generate_facts(self, image_input, custom_vocabulary=None, run_llm_rephrase=False, **kwargs):
        try:
            detected_objects_list, detection_list_dicts, filename_prefix = self._process_common(
                image_input, custom_vocabulary, **kwargs)  # This now uses YOLOE internally
            if not detected_objects_list:
                self.logger.warning("Common processing failed to produce detections for fact generation.")
                return [], [], []

            template_facts = self.fact_prompter.evaluate_predicates_on_pairs(detection_list_dicts)
            rephrased_qas = []
            if run_llm_rephrase and template_facts:
                if not self.llm_pipeline:
                    self.logger.warning("LLM pipeline not initialized. Skipping LLM rephrasing.")
                else:
                    llm_prompts = prepare_llm_prompts_from_facts(template_facts, detection_list_dicts)
                    if llm_prompts: rephrased_qas = self._run_llm_rephrasing_hf(llm_prompts)
            if self.cfg.get("vis", False): self.logger.info(
                f"Wis3D visualization potentially saved for {filename_prefix} in {self.cfg.wis3d_folder}")
            return detected_objects_list, template_facts, rephrased_qas
        except SkipImageException as e:
            self.logger.warning(f"Fact generation skipped for image: {e}"); return [], [], []
        except Exception as e:
            self.logger.error(f"Error during fact generation: {e}", exc_info=True); return [], [], []

    # generate_qa (NO CHANGE in its internal logic other than what _process_common implies)
    def generate_qa(self, image_input, custom_vocabulary=None, **kwargs):
        try:
            detected_objects_list, detection_list_dicts, filename_prefix = self._process_common(
                image_input, custom_vocabulary, **kwargs)  # This now uses YOLOE internally
            if not detected_objects_list:
                self.logger.warning("Common processing failed to produce detections for QA generation.")
                return [], []

            vqa_results = self.qa_prompter.evaluate_predicates_on_pairs(detection_list_dicts)
            template_qas = parse_qas_from_vqa_results(vqa_results)
            if self.cfg.get("vis", False): self.logger.info(
                f"Wis3D visualization potentially saved for {filename_prefix} in {self.cfg.wis3d_folder}")
            return detected_objects_list, template_qas
        except SkipImageException as e:
            self.logger.warning(f"QA generation skipped for image: {e}"); return [], []
        except Exception as e:
            self.logger.error(f"Error during QA generation: {e}", exc_info=True); return [], []

    # _parse_llm_json_output (NO CHANGE)
    def _parse_llm_json_output(self, llm_output_text):
        llm_output_text_stripped = llm_output_text.strip()
        match_json_block = re.search(r"```json\s*(\{.*?\})\s*```", llm_output_text_stripped, re.DOTALL)
        if match_json_block:
            json_str = match_json_block.group(1)
        else:
            first_brace = llm_output_text_stripped.find('{');
            last_brace = llm_output_text_stripped.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
                json_str = llm_output_text_stripped[first_brace: last_brace + 1]
            else:
                if llm_output_text_stripped.startswith("{") and llm_output_text_stripped.endswith("}"):
                    json_str = llm_output_text_stripped
                else:
                    self.logger.warning(
                        f"Could not clearly identify JSON block in LLM output: {llm_output_text_stripped[:200]}"); return None
        try:
            json_str_cleaned = json_str; parsed_json = json.loads(json_str_cleaned); return parsed_json
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"JSONDecodeError for string: '{json_str[:200]}...'. Error: {e}. Original text: {llm_output_text_stripped[:200]}"); return None

    # _run_llm_rephrasing_hf (NO CHANGE)
    def _run_llm_rephrasing_hf(self, llm_prompts):
        if not self.llm_pipeline or not self.llm_tokenizer_hf:
            self.logger.warning("Hugging Face LLM pipeline or tokenizer not available. Skipping rephrasing.")
            return []
        rephrased_conversations = []
        for user_prompt_text in llm_prompts:
            messages = [{"role": "system", "content": LLM_HF_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Input:\n{user_prompt_text}\nOutput:"}]
            try:
                prompt_for_llm = self.llm_tokenizer_hf.apply_chat_template(messages, tokenize=False,
                                                                           add_generation_prompt=True)
            except Exception as e_template:
                self.logger.error(f"Failed to apply chat template: {e_template}. Skipping this prompt.",
                                  exc_info=True); continue

            max_retries = self.cfg.get("llm_max_retries", 3);
            success = False;
            q_final, a_final = None, None
            for attempt in range(max_retries):
                try:
                    terminators = []
                    if self.llm_tokenizer_hf.eos_token_id is not None: terminators.append(
                        self.llm_tokenizer_hf.eos_token_id)
                    im_end_token_id = self.llm_tokenizer_hf.convert_tokens_to_ids("<|im_end|>")
                    if isinstance(im_end_token_id, int) and im_end_token_id not in terminators: terminators.append(
                        im_end_token_id)
                    eot_id_llama = self.llm_tokenizer_hf.convert_tokens_to_ids("<|eot_id|>")
                    if isinstance(eot_id_llama, int) and eot_id_llama not in terminators: terminators.append(
                        eot_id_llama)
                    eos_pipeline_arg = terminators if terminators else None

                    pipeline_args = {"max_new_tokens": self.cfg.get("llm_max_new_tokens", 512),
                                     "temperature": self.cfg.get("llm_temperature", 0.1), "do_sample": True}
                    if eos_pipeline_arg is not None: pipeline_args["eos_token_id"] = eos_pipeline_arg
                    if self.llm_tokenizer_hf.pad_token_id is not None:
                        pipeline_args["pad_token_id"] = self.llm_tokenizer_hf.pad_token_id
                    elif self.llm_tokenizer_hf.eos_token_id is not None:
                        pipeline_args["pad_token_id"] = self.llm_tokenizer_hf.eos_token_id

                    generated_outputs = self.llm_pipeline(prompt_for_llm, **pipeline_args)
                    actual_llm_generation = None
                    if generated_outputs and isinstance(generated_outputs, list) and generated_outputs[0]:
                        if "generated_text" in generated_outputs[0]:
                            full_response_with_prompt = generated_outputs[0]["generated_text"]
                            if isinstance(full_response_with_prompt, str):
                                if full_response_with_prompt.startswith(prompt_for_llm):
                                    actual_llm_generation = full_response_with_prompt[len(prompt_for_llm):]
                                else:
                                    actual_llm_generation = full_response_with_prompt

                    if actual_llm_generation is not None and isinstance(actual_llm_generation, str):
                        for known_eos in ["<|im_end|>", "<|eot_id|>", self.llm_tokenizer_hf.eos_token]:
                            if known_eos and actual_llm_generation.strip().endswith(known_eos):
                                actual_llm_generation = actual_llm_generation.strip()[:-len(known_eos)].strip()
                        json_response = self._parse_llm_json_output(actual_llm_generation)
                        if not json_response: self.logger.warning(
                            f"LLM attempt {attempt + 1}: Could not parse JSON: {actual_llm_generation[:200]}..."); continue
                    else:
                        self.logger.warning(
                            f"LLM attempt {attempt + 1}: actual_llm_generation issue. Output: {str(generated_outputs)[:200]}"); continue

                    question, answer = json_response.get("Question"), json_response.get("Answer")
                    if question is None or answer is None: self.logger.warning(
                        f"LLM response missing Q or A. Parsed: {json_response}"); continue

                    question = question[2:] if question.startswith(". ") else question
                    answer = answer[2:] if answer.startswith(". ") else answer
                    prompt_tags = set(re.findall(r"<region\d+>", user_prompt_text))
                    question_tags = set(re.findall(r"<region\d+>", question))

                    if prompt_tags.issubset(question_tags) and question_tags.issubset(prompt_tags):
                        if all(question.count(tag) == 1 for tag in prompt_tags):
                            q_final, a_final = question, answer;
                            success = True;
                            break
                        else:
                            self.logger.debug(
                                f"LLM attempt {attempt + 1}: <regionX> >1 times. Prompt: '{user_prompt_text[:50]}...'")
                    else:
                        self.logger.debug(
                            f"LLM attempt {attempt + 1}: <regionX> mismatch. Prompt: '{user_prompt_text[:50]}...'")
                except Exception as e:
                    self.logger.warning(
                        f"LLM rephrase attempt {attempt + 1} for '{user_prompt_text[:50]}...' failed: {e}",
                        exc_info=True)

            if success:
                rephrased_conversations.append((q_final, a_final)); self.logger.info(
                    f"LLM Rephrased => Q: {q_final} || A: {a_final}")
            else:
                self.logger.warning(
                    f"LLM failed for prompt: '{user_prompt_text[:100]}...' after {max_retries} attempts.")
        return rephrased_conversations

    # __del__ (NO CHANGE)
    def __del__(self):
        if hasattr(self, 'llm_pipeline') and self.llm_pipeline is not None:
            if hasattr(self.llm_pipeline, 'model') and self.llm_pipeline.model is not None:
                if hasattr(self.llm_pipeline.model, 'cpu'):
                    try:
                        self.llm_pipeline.model.cpu(); self.logger.info("Moved LLM model to CPU.")
                    except Exception as e:
                        self.logger.warning(f"Could not move LLM model to CPU: {e}")
                del self.llm_pipeline.model
            del self.llm_pipeline
            self.llm_pipeline = None;
            self.llm_tokenizer_hf = None
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.logger.info("Attempted to release LLM resources.")


# --- Example Usage (Optimized for RPi) ---
if __name__ == "__main__":
    import sys
    import logging  # For dummy logger if osdsynth not available

    # CONFIG FOR RASPBERRY PI
    # Create this file in a 'configs' directory or adjust path
    CONFIG_FILE_NAME = "rpi_config.py"
    config_dir = "configs"
    config_file_path = os.path.join(config_dir, CONFIG_FILE_NAME)

    # Create dummy config if not exists for testing
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    if not os.path.exists(config_file_path):
        print(f"WARNING: RPi config '{config_file_path}' not found. Creating a dummy one.")
        with open(config_file_path, "w") as f:
            f.write("# Dummy RPi Config (customize this!)\n")
            f.write("log_dir = './temp_outputs_rpi/log'\n")
            f.write("vis = False\n")
            f.write("image_resize_height = 320\n")  # Smaller images
            f.write("yoloe_model_path = 'yolov8n-seg.pt' # Use a NANO model if available for YOLOE or YOLOv8\n")
            f.write("yoloe_input_size = 320\n")
            f.write("yoloe_confidence_threshold = 0.45\n")
            f.write("min_mask_area_pixel = 150\n")
            f.write("yoloe_mask_dilate_iterations = 0\n")
            f.write("enable_unik3d = True # Set to False to disable 3D path for max speed\n")
            f.write("pcd_enable_sor = False\n")
            f.write("pcd_voxel_size = 0.05\n")
            f.write("dbscan_remove_noise = False\n")
            f.write("obb_robust = False\n")
            f.write("min_points_threshold = 10\n")
            f.write("min_points_threshold_after_denoise = 5\n")
            f.write("enable_llm = False # LLM disabled for RPi\n")
            f.write("llm_model_name_hf = None\n")
            f.write("crop_padding = 5\n")  # Smaller padding for crops
            f.write("sort_detections_by_area = False # Skip sorting for minor speedup\n")
            f.write("vis_global_scene = False # Don't visualize global scene in Wis3D\n")
        print(
            f"Created dummy config: {config_file_path}. Please review and customize it, especially 'yoloe_model_path'.")

    demo_image_dir = "./demo_images"
    if not os.path.exists(demo_image_dir): os.makedirs(demo_image_dir)
    # Use a common image, e.g., from COCO or a simple scene
    demo_image_path = os.path.join(demo_image_dir, "indoor.png")  # Example from Ultralytics assets
    # Or your "indoor.png"

    if not os.path.exists(demo_image_path):
        print(f"Warning: Demo image {demo_image_path} not found. Creating a dummy one.")
        try:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(dummy_img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green box
            cv2.rectangle(dummy_img, (300, 200), (450, 350), (0, 0, 255), -1)  # Red box
            cv2.imwrite(demo_image_path, dummy_img)
            print(f"Created dummy image: {demo_image_path}")
        except Exception as e:
            print(f"Could not create dummy image: {e}")

    print(f"--- Initializing SGG for RPi with YOLOE ---")
    print(f"Using Config: {config_file_path}")
    print(f"Device: CPU (forced)")

    main_total_time_start = time.time()
    generator_main = None
    try:
        # For RPi, LLM is typically off. Set llm_model_name_hf=None directly.
        generator_main = GeneralizedSceneGraphGenerator(
            config_path=config_file_path,
            device="cpu",  # Explicitly CPU
            llm_model_name_hf=None  # LLM off by default for RPi init
        )
    except Exception as e_init:
        print(f"FATAL ERROR during generator initialization: {e_init}")
        import traceback;

        traceback.print_exc()
        exit()

    current_image_to_process = demo_image_path
    if not os.path.exists(current_image_to_process):
        print(f"ERROR: Image to process '{current_image_to_process}' not found. Exiting.")
        exit()

    print(f"\n--- Generating Facts for {current_image_to_process} (RPi Optimized) ---")
    try:
        # custom_vocab = ["person", "car"] # Example: YOLOE will try these if its model knows them
        custom_vocab = None  # Detect all classes YOLOE model knows

        # LLM rephrase is False by default via enable_llm=False in RPi config
        # Override here if you want to test it AND enable_llm=True in config
        run_llm_flag = generator_main.cfg.get("enable_llm", False) and \
                       (generator_main.llm_pipeline is not None)

        detected_objects_list, facts, rephrased_qas_f = generator_main.generate_facts(
            current_image_to_process,
            custom_vocabulary=custom_vocab,
            run_llm_rephrase=run_llm_flag,
            vis=generator_main.cfg.get("vis", False)  # Usually False for RPi
        )

        if detected_objects_list:
            print(f"\nSUCCESS: Generated {len(detected_objects_list)} DetectedObject instances:")
            for i, obj in enumerate(detected_objects_list[:3]):  # Print first 3 for brevity
                print(f"  Object {i + 1}: {obj}")
            if len(detected_objects_list) > 3: print("  ...")

            print(f"\nGenerated {len(facts)} template facts:")
            for i, fact_str in enumerate(facts[:3]):  # Print first 3
                print(f"  Fact {i + 1}: {fact_str}")
            if len(facts) > 3: print("  ...")

            if rephrased_qas_f:
                print(f"\nGenerated {len(rephrased_qas_f)} LLM-rephrased QAs from facts:")
                for i_qa, (q, a) in enumerate(rephrased_qas_f[:2]):  # Print first 2
                    print(f"  LLM QA {i_qa + 1}: Q: {q[:60]}... || A: {a[:60]}...")
                if len(rephrased_qas_f) > 2: print("  ...")
            elif run_llm_flag:
                print("LLM rephrasing was enabled, but no QAs generated (no facts or LLM failed).")
        else:
            print("No DetectedObject instances generated (no valid detections or processing failed).")

    except SkipImageException as e_skip:
        print(f"Skipped image processing: {e_skip}")
    except Exception as e_facts:
        print(f"Error in fact generation example: {e_facts}")
        import traceback;

        traceback.print_exc()

    # --- Optional: Generate QAs (simpler path, no LLM rephrase involved here) ---
    # print(f"\n--- Generating QAs for {current_image_to_process} (RPi Optimized) ---")
    # try:
    #     detected_objects_list_qa, template_qas_list = generator_main.generate_qa(
    #         current_image_to_process,
    #         custom_vocabulary=custom_vocab,
    #         vis=generator_main.cfg.get("vis", False)
    #     )
    #     if detected_objects_list_qa:
    #         print(f"\nSUCCESS (QA Path): Generated {len(detected_objects_list_qa)} DetectedObject instances.")
    #         print(f"Generated {len(template_qas_list)} template QAs:")
    #         for i, qa_pair in enumerate(template_qas_list[:3]):
    #             print(f"  QA {i+1}: Q: {qa_pair[0][:60]}... || A: {qa_pair[1][:60]}...")
    #         if len(template_qas_list) > 3: print("  ...")
    #     else:
    #         print("(QA Path) No DetectedObject instances generated.")
    # except SkipImageException as e_skip_qa:
    #     print(f"(QA Path) Skipped image processing: {e_skip_qa}")
    # except Exception as e_qa_gen:
    #     print(f"Error in QA generation example: {e_qa_gen}")
    #     import traceback; traceback.print_exc()

    if generator_main:
        del generator_main  # Call __del__ for cleanup

    gc.collect()  # Force garbage collection

    main_total_time_end = time.time()
    print(f"\n--- Total script execution time: {main_total_time_end - main_total_time_start:.2f} seconds ---")
    print("\nProcessing complete.")