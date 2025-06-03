import streamlit as st
from streamlit_drawable_canvas import st_canvas
import copy
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add spatialrgpt main path
# Assuming streamlit_app.py is in a subfolder like 'apps/' and project root is parent.
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from utils.markdown import process_markdown  # Assuming this exists and works
    # from utils.sam_utils import get_box_inputs # Not directly used; parsing canvas output instead
    from utils.som import draw_mask_and_number_on_image

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, process_regions, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    st.error(
        f"Failed to import necessary modules from MealsRetrieval project. Ensure correct project structure and PYTHONPATH. Error: {e}")
    st.stop()

# --- Define default colors for markdown ---
# These colors will be passed as 'color_history' to process_markdown
# (Ensure this list provides enough colors for expected highlighted segments)
DEFAULT_MARKDOWN_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 0), (255, 0, 128), (0, 255, 128),
    (128, 255, 0), (0, 128, 255), (128, 0, 255), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (128, 255, 255), (255, 128, 255)
]

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitl'

# --- Environment Variable Checks ---
DEPTH_ANYTHING_PATH = os.environ.get("DEPTH_ANYTHING_PATH")
SAM_CKPT_PATH = os.environ.get("SAM_CKPT_PATH")


# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner=False)
def load_depth_predictor_cached():
    if not DEPTH_ANYTHING_PATH:
        st.error("Error: DEPTH_ANYTHING_PATH environment variable not set.")
        return None, None
    if not os.path.isdir(DEPTH_ANYTHING_PATH):
        st.error(f"Error: DEPTH_ANYTHING_PATH ({DEPTH_ANYTHING_PATH}) is not a valid directory.")
        return None, None

    # Temporarily add DepthAnything to sys.path for its internal imports
    original_sys_path = list(sys.path)
    if DEPTH_ANYTHING_PATH not in sys.path:
        sys.path.insert(0, DEPTH_ANYTHING_PATH)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
        from torchvision.transforms import Compose

        depth_model_path = os.path.join(DEPTH_ANYTHING_PATH, "checkpoints", "depth_anything_v2_vitl.pth")
        if not os.path.isfile(depth_model_path):
            st.error(f"Error: Depth model checkpoint not found at {depth_model_path}")
            return None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        depth_model = DepthAnythingV2(**model_configs[encoder])

        depth_model.load_state_dict(torch.load(depth_model_path, map_location=device))
        depth_model = depth_model.to(device).eval()

        depth_transform = Compose(
            [
                Resize(
                    width=518, height=518, resize_target=False, keep_aspect_ratio=True,
                    ensure_multiple_of=14, resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        # st.success("Depth model loaded successfully!")
        return depth_model, depth_transform
    except Exception as e:
        st.error(f"Error loading Depth model: {e}")
        return None, None
    finally:
        # Restore original sys.path
        sys.path = original_sys_path


@st.cache_resource(show_spinner=False)
def load_sam_predictor_cached():
    if not SAM_CKPT_PATH:
        st.error("Error: SAM_CKPT_PATH environment variable not set.")
        return None
    if not os.path.isfile(SAM_CKPT_PATH):
        st.error(f"Error: SAM checkpoint not found at {SAM_CKPT_PATH}")
        return None

    try:
        from segment_anything_hq import SamPredictor, sam_model_registry
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT_PATH).to(device).eval()
        sam_predictor = SamPredictor(sam)
        # st.success("SAM model loaded successfully!")
        return sam_predictor
    except Exception as e:
        st.error(f"Error loading SAM model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_spatialrgpt_model_cached(model_path, model_name):
    if not model_path or not model_name:
        st.error("Model path or model name not provided for MealsRetrieval.")
        return None, None, None, None
    # if not os.path.isdir(model_path): # model_path is usually a directory
    #     st.error(f"MealsRetrieval model path does not exist or is not a directory: {model_path}")
    #     return None, None, None, None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # load_pretrained_model might handle device mapping, or pass device explicitly if supported
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_name, device_map=None  # Let builder handle device map or set explicitly
        )
        model = model.to(device).eval()
        # st.success(f"MealsRetrieval model '{model_name}' loaded successfully!")
        return tokenizer, model, image_processor, context_len
    except Exception as e:
        st.error(f"Error loading MealsRetrieval model '{model_name}': {e}")
        return None, None, None, None


# --- Core Logic Functions (Adapted from cli.py) ---
def segment_using_boxes_st(raw_image, boxes, sam_predictor, use_segmentation=True):
    orig_h, orig_w = raw_image.shape[:2]
    bboxes_np = np.array(boxes)
    seg_masks = []  # Full resolution masks for VLM

    # Create a display image, resized for consistent visualization
    # Aspect ratio preserving resize to fit width 640 (common display size)
    display_width = 640
    display_height = int(orig_h * (display_width / orig_w))
    display_image_resized = cv2.resize(raw_image, (display_width, display_height))

    if bboxes_np.size == 0:
        return [], display_image_resized  # No boxes, return empty masks and original (resized) image

    if use_segmentation:
        if sam_predictor is None:
            st.error("SAM predictor not loaded. Cannot perform segmentation.")
            return [], display_image_resized
        sam_predictor.set_image(raw_image)  # Set image for SAM
        for bbox in bboxes_np:
            # SAM expects box as [x1, y1, x2, y2]
            masks_sam, scores, _ = sam_predictor.predict(box=bbox, multimask_output=True)
            seg_masks.append(masks_sam[np.argmax(scores)].astype(np.uint8))
    else:  # Use bounding boxes directly as masks
        for bbox in bboxes_np:
            zero_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            zero_mask[y1:y2, x1:x2] = 1
            seg_masks.append(zero_mask)

    segmented_display_image = display_image_resized.copy()  # Default to resized original if no masks
    if len(seg_masks) > 0:
        region_labels = [f"Region {i}" for i in range(len(seg_masks))]
        # draw_mask_and_number_on_image can handle masks of different resolution than the image
        # it draws on, by resizing them internally.
        segmented_display_image = draw_mask_and_number_on_image(
            display_image_resized.copy(), seg_masks, region_labels,
            label_mode="1", alpha=0.5, anno_mode=["Mask", "Mark", "Box"]
        )
    return seg_masks, segmented_display_image


def get_depth_map_st(raw_image, depth_model, depth_transform):
    if depth_model is None or depth_transform is None:
        st.error("Depth model/transform not loaded. Cannot generate depth map.")
        return np.zeros_like(raw_image, dtype=np.uint8)

    orig_h, orig_w = raw_image.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare image for depth model
    depth_input_image_dict = depth_transform({"image": raw_image / 255.0})  # transform expects dict
    depth_input_image = depth_input_image_dict["image"]
    depth_input_image = torch.from_numpy(depth_input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_depth = depth_model(depth_input_image)

    raw_depth = F.interpolate(raw_depth[None], (orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0]
    raw_depth = raw_depth.cpu().numpy()  # Move to CPU for numpy operations

    # Normalize and convert to 8-bit
    min_val, max_val = raw_depth.min(), raw_depth.max()
    if max_val - min_val > 1e-6:  # Avoid division by zero if depth is flat
        raw_depth = (raw_depth - min_val) / (max_val - min_val) * 255.0
    else:
        raw_depth = np.zeros_like(raw_depth)  # Or set to a constant value
    raw_depth = raw_depth.astype(np.uint8)
    colorized_depth = cv2.cvtColor(raw_depth, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel for display & VLM
    return colorized_depth


def inference_vlm_st(
        input_str: str, raw_image: np.ndarray, seg_masks: list, colorized_depth: np.ndarray,
        tokenizer, spatialrgpt_model, image_processor,
        current_conv_state, current_conv_history,
        conv_mode: str, use_depth: bool, use_bfloat: bool,
        follow_up: bool, temperature: float, max_new_tokens: int
):
    # Determine base query by replacing <regionX>
    if use_depth:
        query_base = re.sub(r"<region\d+>", "<mask> <depth>", input_str)
    else:
        query_base = re.sub(r"<region\d+>", "<mask>", input_str)

    # Manage conversation state
    active_conv = current_conv_state
    active_conv_history = current_conv_history

    if not follow_up or active_conv is None:
        active_conv = conv_templates[conv_mode].copy()
        if not follow_up:  # New conversation explicitly started
            active_conv_history = {"user": [], "model": []}
            st.info("Starting a new conversation.")
        query = DEFAULT_IMAGE_TOKEN + "\n" + query_base
    else:  # Is a follow-up
        query = query_base

    active_conv_history["user"].append(input_str)  # Store original user input

    active_conv.append_message(active_conv.roles[0], query)
    active_conv.append_message(active_conv.roles[1], None)  # Placeholder for model's response
    prompt = active_conv.get_prompt()

    # Prepare image/depth tensors
    pil_raw_image = Image.fromarray(raw_image)
    pil_colorized_depth = Image.fromarray(colorized_depth)

    device = spatialrgpt_model.device
    selected_dtype = torch.float16  # Default to float16
    if use_bfloat and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        selected_dtype = torch.bfloat16
    elif use_bfloat and torch.cuda.is_available():
        st.warning("bfloat16 selected but not supported on this GPU. Using float16.")

    # Temporarily cast model for inference if needed (as in cli.py)
    original_model_dtype = next(spatialrgpt_model.parameters()).dtype
    if original_model_dtype != selected_dtype and str(device) != 'cpu':  # Avoid dtype change on CPU
        spatialrgpt_model.to(dtype=selected_dtype)

    images_tensor = process_images([pil_raw_image], image_processor, spatialrgpt_model.config).to(device,
                                                                                                  dtype=selected_dtype if str(
                                                                                                      device) != 'cpu' else torch.float32)
    depths_tensor = process_images([pil_colorized_depth], image_processor, spatialrgpt_model.config).to(device,
                                                                                                        dtype=selected_dtype if str(
                                                                                                            device) != 'cpu' else torch.float32)

    # Mask processing based on *current* input_str's regions
    current_input_region_tags = re.findall(r"<region(\d+)>", input_str)
    current_input_region_indices_int = [int(tag) for tag in current_input_region_tags]

    final_masks_for_model = None
    if len(seg_masks) > 0 and len(current_input_region_indices_int) > 0:
        # Original line that creates PIL images:
        # pil_seg_masks_for_processing = [Image.fromarray(m * 255) for m in seg_masks]

        # WORKAROUND: Convert masks to NumPy arrays before passing to process_regions.
        # This assumes that process_regions in llava.mm_utils.py expects NumPy arrays
        # for its cv2.resize call, or that subsequent operations are NumPy compatible
        # if it was expecting PIL images but had a bug in the resize call.
        # Each element 'm' in seg_masks is already a NumPy array (0 or 1).
        # Image.fromarray(m * 255) converts it to a PIL Image (mode 'L').
        # np.array(...) converts this PIL Image back to a NumPy array (uint8, 0 or 255 values).
        np_masks_for_processing = [np.array(Image.fromarray(m * 255)) for m in seg_masks]

        _masks_tensor_all_available = process_regions(
            np_masks_for_processing,  # Pass NumPy arrays instead of PIL Images
            image_processor,
            spatialrgpt_model.config
        ).to(device, dtype=selected_dtype if str(device) != 'cpu' else torch.float32)

        actual_mask_indices_to_pass_to_model = []
        for r_idx in current_input_region_indices_int:  # Iterate through regions mentioned in current prompt
            if 0 <= r_idx < _masks_tensor_all_available.size(0):
                actual_mask_indices_to_pass_to_model.append(r_idx)
            else:
                st.warning(
                    f"Region index {r_idx} in current prompt is out of bounds for available masks ({_masks_tensor_all_available.size(0)} total). It will be ignored.")

        if actual_mask_indices_to_pass_to_model:
            final_masks_for_model = _masks_tensor_all_available[actual_mask_indices_to_pass_to_model]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    stop_str = active_conv.sep if active_conv.sep_style != SeparatorStyle.TWO else active_conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = spatialrgpt_model.generate(
            input_ids,
            images=[images_tensor],
            depths=[depths_tensor],
            masks=[final_masks_for_model] if final_masks_for_model is not None else None,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # Restore model dtype if changed
    if original_model_dtype != selected_dtype and str(device) != 'cpu':
        spatialrgpt_model.to(dtype=original_model_dtype)

    outputs_raw = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs_stripped = outputs_raw.strip()
    if outputs_stripped.endswith(stop_str):
        outputs_stripped = outputs_stripped[:-len(stop_str)]
    outputs_final_model = outputs_stripped.strip()

    # Remap output region indices
    mapping_dict = {str(out_idx): str(in_idx_str) for out_idx, in_idx_str in enumerate(current_input_region_tags)}
    remapped_outputs = outputs_final_model
    if mapping_dict:
        try:
            remapped_outputs = re.sub(r"\[([0-9]+)\]", lambda x: f"[{mapping_dict[x.group(1)]}]" if x.group(
                1) in mapping_dict else x.group(0), outputs_final_model)
        except KeyError as e:  # Should be caught by check in lambda
            st.warning(
                f"Output remapping failed: Model referred to index {e} not in current prompt's regions. Raw output shown.")

    # Update conversation state
    if active_conv.messages and active_conv.messages[-1][0] == active_conv.roles[1] and active_conv.messages[-1][
        1] is None:
        active_conv.messages.pop()
    active_conv.append_message(active_conv.roles[1], outputs_raw)  # Store raw model output
    active_conv_history["model"].append(remapped_outputs)  # Store remapped for display

    return remapped_outputs, active_conv, active_conv_history


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="MealsRetrieval Demo", initial_sidebar_state="collapsed")
st.title("✨ MealsRetrieval Interactive Demo ✨")

# --- Load Models ---
# Attempt to load models. Errors are shown via st.error within loading functions.
# These are @st.cache_resource, so they run once per session.
# with st.spinner("Loading segmentation and depth models..."):
with st.spinner("Loading miscellaneous models..."):
    depth_model, depth_transform = load_depth_predictor_cached()
    sam_predictor = load_sam_predictor_cached()

# Sidebar for MealsRetrieval model path and inference parameters
st.sidebar.header("MealsRetrieval Configuration")
MODEL_PATH_DEFAULT = os.environ.get("SPATIALRGPT_MODEL_PATH", "a8cheng/SpatialRGPT-VILA1.5-8B")  # Adjust as needed
MODEL_NAME_DEFAULT = os.environ.get("SPATIALRGPT_MODEL_NAME", "SpatialRGPT-VILA1.5-8B")  # Adjust

model_path_st = st.sidebar.text_input("MealsRetrieval Model Path", value=MODEL_PATH_DEFAULT)
model_name_st = st.sidebar.text_input("MealsRetrieval Model Name", value=MODEL_NAME_DEFAULT)

# with st.spinner(f"Loading MealsRetrieval model '{model_name_st}'..."):
with st.spinner(f"Loading MealsRetrieval model..."):
    tokenizer, spatialrgpt_model, image_processor, context_len = load_spatialrgpt_model_cached(model_path_st,
                                                                                               model_name_st)

all_models_loaded = all([depth_model, depth_transform, sam_predictor, tokenizer, spatialrgpt_model, image_processor])
if not all_models_loaded:
    st.error(
        "One or more critical models failed to load. Please check configurations and environment variables (see sidebar). The application may not function correctly.")
    # st.stop() # Uncomment to halt if any model fails

st.sidebar.header("Inference Parameters")
# Find default conv_mode index, robustly
available_conv_modes = list(conv_templates.keys())
default_conv_idx = available_conv_modes.index('llama_3') if 'llama_3' in available_conv_modes else 0
conv_mode_st = st.sidebar.selectbox("Conversation Mode", options=available_conv_modes, index=default_conv_idx)

use_sam_st = st.sidebar.checkbox("Use SAM for Segmentation", value=True)
use_depth_st = st.sidebar.checkbox("Use Depth Information (<depth> token)", value=True)
use_bfloat_st = st.sidebar.checkbox("Use bfloat16 (if GPU supports)", value=True)
temperature_st = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
max_new_tokens_st = st.sidebar.slider("Max New Tokens", min_value=64, max_value=2048, value=512, step=64)

# Initialize session state variables
if 'conv' not in st.session_state: st.session_state.conv = None
if 'conv_history' not in st.session_state: st.session_state.conv_history = {"user": [], "model": []}
if 'raw_image' not in st.session_state: st.session_state.raw_image = None
if 'seg_masks' not in st.session_state: st.session_state.seg_masks = []
if 'segmented_display_image' not in st.session_state: st.session_state.segmented_display_image = None
if 'colorized_depth_map' not in st.session_state: st.session_state.colorized_depth_map = None
if 'drawn_boxes' not in st.session_state: st.session_state.drawn_boxes = []
if 'uploaded_file_id' not in st.session_state: st.session_state.uploaded_file_id = None

# --- Main UI Layout ---
uploaded_file = st.file_uploader("1. Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.file_id != st.session_state.uploaded_file_id:  # New file uploaded
        try:
            pil_image = Image.open(uploaded_file).convert('RGB')
            st.session_state.raw_image = np.array(pil_image)
            st.session_state.uploaded_file_id = uploaded_file.file_id  # Store new file ID

            # Reset states for new image
            st.session_state.drawn_boxes = []
            st.session_state.seg_masks = []
            st.session_state.segmented_display_image = None
            st.session_state.colorized_depth_map = None  # Depth map is image-specific
            st.session_state.conv = None  # Reset conversation for new image
            st.session_state.conv_history = {"user": [], "model": []}
            st.info("New image uploaded. Segmentation, depth, and conversation history reset.")
        except Exception as e:
            st.error(f"Failed to load image: {e}")
            st.session_state.raw_image = None  # Ensure reset on failure

if st.session_state.raw_image is not None:
    current_raw_image = st.session_state.raw_image
    img_h_orig, img_w_orig = current_raw_image.shape[:2]

    # --- Canvas for drawing boxes ---
    st.subheader("2. Draw Bounding Boxes (Optional)")
    # Scale image for canvas display if too large, maintaining aspect ratio
    CANVAS_MAX_W, CANVAS_MAX_H = 700, 500
    display_w, display_h = img_w_orig, img_h_orig
    if img_w_orig > CANVAS_MAX_W:
        display_w = CANVAS_MAX_W
        display_h = int(img_h_orig * (CANVAS_MAX_W / img_w_orig))
    if display_h > CANVAS_MAX_H:  # Check height again if previous scaling made it too tall
        display_h = CANVAS_MAX_H
        display_w = int(img_w_orig * (CANVAS_MAX_H / img_h_orig))

    scale_x_canvas = img_w_orig / display_w
    scale_y_canvas = img_h_orig / display_h

    canvas_bg_pil = Image.fromarray(current_raw_image).resize((display_w, display_h))

    col_canvas, col_display = st.columns(2)
    with col_canvas:
        st.markdown("Draw rectangles on the image below. Each rectangle will be a numbered region.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)", stroke_width=2, stroke_color="#FF8C00",  # DarkOrange
            background_image=canvas_bg_pil, update_streamlit=True,
            height=display_h, width=display_w, drawing_mode="rect", key="canvas"
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            new_drawn_boxes_scaled = []
            for obj in objects:
                if obj["type"] == "rect":
                    x1 = int(obj["left"] * scale_x_canvas)
                    y1 = int(obj["top"] * scale_y_canvas)
                    x2 = int((obj["left"] + obj["width"]) * scale_x_canvas)
                    y2 = int((obj["top"] + obj["height"]) * scale_y_canvas)
                    new_drawn_boxes_scaled.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

            # If boxes changed, re-segment and update display
            if new_drawn_boxes_scaled != st.session_state.drawn_boxes:
                st.session_state.drawn_boxes = new_drawn_boxes_scaled
                if sam_predictor and current_raw_image is not None:  # Ensure dependencies are met
                    with st.spinner("Segmenting regions..."):
                        st.session_state.seg_masks, st.session_state.segmented_display_image = \
                            segment_using_boxes_st(current_raw_image, st.session_state.drawn_boxes, sam_predictor,
                                                   use_sam_st)
                    st.rerun()  # Rerun to update the display image immediately

    with col_display:
        st.markdown("**Segmented Regions View**")
        if st.session_state.segmented_display_image is not None:
            st.image(st.session_state.segmented_display_image, caption="Numbered Regions", use_container_width=True)
        else:  # Show original (resized for canvas) if no segmentation yet
            st.image(canvas_bg_pil, caption="Original Image (Draw boxes on the left)", use_container_width=True)

        if use_depth_st:
            st.markdown("**Depth Map View**")
            if st.session_state.colorized_depth_map is None and depth_model and depth_transform:  # Auto-generate if needed and models are ready
                with st.spinner("Generating depth map..."):
                    st.session_state.colorized_depth_map = get_depth_map_st(current_raw_image, depth_model,
                                                                            depth_transform)
                    st.rerun()  # Rerun to show depth map

            if st.session_state.colorized_depth_map is not None:
                st.image(st.session_state.colorized_depth_map, caption="Colorized Depth Map", use_container_width=True)
            else:
                st.caption(
                    "Depth map will be generated when 'Use Depth Information' is checked and a prompt is sent, or if auto-generation failed.")

    # --- Prompt and Inference ---
    st.subheader("3. Chat with MealsRetrieval")

    # Display conversation history
    if st.session_state.conv_history["user"]:
        for i in range(len(st.session_state.conv_history["user"])):
            st.markdown(f"👤 **You:** {st.session_state.conv_history['user'][i]}")
            if i < len(st.session_state.conv_history["model"]):
                # Using process_markdown
                try:
                    # CORRECTED CALL to process_markdown:
                    # Pass st.session_state.conv_history['model'][i] as output_str
                    # Pass DEFAULT_MARKDOWN_COLORS as color_history
                    html_output = process_markdown(
                        st.session_state.conv_history['model'][i],
                        DEFAULT_MARKDOWN_COLORS  # Use the predefined color list
                    )
                    st.markdown(f"🤖 **MealsRetrieval:** {html_output}", unsafe_allow_html=True)
                except NameError as ne:  # Specifically if process_markdown itself isn't found
                    st.error(f"Markdown processing function not found: {ne}")
                    st.markdown(f"🤖 **MealsRetrieval (raw):** {st.session_state.conv_history['model'][i]}")
                except Exception as e:  # Catch other errors from process_markdown
                    st.error(f"Error during markdown processing: {e}")
                    st.markdown(f"🤖 **MealsRetrieval (raw):** {st.session_state.conv_history['model'][i]}")
        st.markdown("---")

    follow_up_st_ui = st.checkbox("Follow-up to previous turn", value=True if st.session_state.conv else False,
                                  disabled=not st.session_state.conv)
    prompt_text = st.text_area(
        "Enter your prompt (e.g., 'Describe <region0>', 'Is <region1> to the left of <region2>?')", height=100,
        key="prompt_input")

    if st.button("💬 Send Prompt", disabled=(not all_models_loaded or not prompt_text.strip())):
        if not prompt_text.strip():
            st.warning("Please enter a prompt.")
        elif not st.session_state.drawn_boxes and re.search(r"<region\d+>", prompt_text):
            st.warning(
                "Your prompt mentions regions (e.g., <region0>), but no boxes are drawn. Please draw boxes first or remove region tags from your prompt.")
        elif all_models_loaded:
            with st.spinner("🧠 MealsRetrieval is thinking..."):
                # Ensure depth map is available if use_depth_st is true
                current_depth_map_for_vlm = np.zeros_like(current_raw_image, dtype=np.uint8)  # Default blank
                if use_depth_st:
                    if st.session_state.colorized_depth_map is None and depth_model and depth_transform:
                        st.session_state.colorized_depth_map = get_depth_map_st(current_raw_image, depth_model,
                                                                                depth_transform)
                    if st.session_state.colorized_depth_map is not None:
                        current_depth_map_for_vlm = st.session_state.colorized_depth_map

                # Ensure seg_masks are current (should be handled by canvas update logic, but double check)
                current_seg_masks = st.session_state.seg_masks
                if not current_seg_masks and st.session_state.drawn_boxes and sam_predictor:
                    current_seg_masks, _ = segment_using_boxes_st(
                        current_raw_image, st.session_state.drawn_boxes, sam_predictor, use_sam_st
                    )

                # Call inference
                response_text, updated_conv, updated_conv_history = inference_vlm_st(
                    input_str=prompt_text, raw_image=current_raw_image,
                    seg_masks=current_seg_masks, colorized_depth=current_depth_map_for_vlm,
                    tokenizer=tokenizer, spatialrgpt_model=spatialrgpt_model, image_processor=image_processor,
                    current_conv_state=st.session_state.conv, current_conv_history=st.session_state.conv_history,
                    conv_mode=conv_mode_st, use_depth=use_depth_st, use_bfloat=use_bfloat_st,
                    follow_up=follow_up_st_ui, temperature=temperature_st, max_new_tokens=max_new_tokens_st
                )

                st.session_state.conv = updated_conv
                st.session_state.conv_history = updated_conv_history
                st.rerun()  # Rerun to display the new message and clear prompt

    if st.session_state.conv and st.button("🧹 Clear Conversation History"):
        st.session_state.conv = None
        st.session_state.conv_history = {"user": [], "model": []}
        st.info("Conversation history cleared.")
        st.rerun()

elif not all_models_loaded and any([DEPTH_ANYTHING_PATH, SAM_CKPT_PATH, model_path_st, model_name_st]):
    st.warning(
        "Models are still loading or some paths might be incorrect. Please check the messages above and ensure all configurations in the sidebar are correct and environment variables are set.")
else:
    st.info("👋 Welcome! Please upload an image to begin.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Ensure `DEPTH_ANYTHING_PATH` and `SAM_CKPT_PATH` environment variables are set correctly. Provide the MealsRetrieval model path and name above.")