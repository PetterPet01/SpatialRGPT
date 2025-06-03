#!/usr/bin/env python3
import argparse
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
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)

    from utils.markdown import process_markdown  # Available, but not used for CLI output
    # from utils.sam_utils import get_box_inputs # Not directly used; boxes parsed from args
    from utils.som import draw_mask_and_number_on_image

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, process_regions, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    print(
        f"Failed to import necessary modules from SpatialRGPT project. Ensure correct project structure and PYTHONPATH. Error: {e}")
    sys.exit(1)

# --- Define model configs for DepthAnythingV2 (Module-level constants) ---
MODEL_CONFIGS_DA_V2 = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},  # This is used
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
DEPTH_ENCODER_TYPE = 'vitl'  # Using ViT-L for depth as in streamlit_demo


def get_depth_predictor(device_str, depth_anything_root_path, model_configs, encoder_type):
    # Temporarily add DepthAnythingV2 to sys.path for its internal imports
    original_sys_path = list(sys.path)
    if depth_anything_root_path not in sys.path:
        sys.path.insert(0, depth_anything_root_path)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
        from torchvision.transforms import Compose

        depth_model_path = os.path.join(depth_anything_root_path, "checkpoints",
                                        f"depth_anything_v2_{encoder_type}.pth")
        if not os.path.isfile(depth_model_path):
            print(f"Error: Depth model checkpoint not found at {depth_model_path}")
            return None, None

        depth_model = DepthAnythingV2(**model_configs[encoder_type])
        depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'))  # Load to CPU first
        depth_model = depth_model.to(device_str).eval()

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
        print("DepthAnythingV2 model successfully loaded!")
        return depth_model, depth_transform
    except Exception as e:
        print(f"Error loading DepthAnythingV2 model: {e}")
        return None, None
    finally:
        # Restore original sys.path
        sys.path = original_sys_path


def get_sam_predictor(device_str, sam_checkpoint_path):
    try:
        from segment_anything_hq import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to(device_str).eval()
        predictor = SamPredictor(sam)
        print("SAM model successfully loaded!")
        return predictor
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return None


def segment_using_boxes(raw_image, boxes, use_segmentation, sam_predictor_instance):
    """
    Segments regions in an image using bounding boxes.
    Returns binary (0/1) numpy masks and a display image with masks drawn.
    """
    orig_h, orig_w = raw_image.shape[:2]
    bboxes_np = np.array(boxes)
    seg_masks = []  # List of binary numpy masks (H_orig, W_orig)

    # Create a display image, resized for consistent visualization for saving
    display_width = 640
    display_height = int(orig_h * (display_width / orig_w))
    display_image_resized = cv2.resize(raw_image, (display_width, display_height))

    if bboxes_np.size == 0:
        return [], display_image_resized  # No boxes, return empty masks

    if use_segmentation:
        if sam_predictor_instance is None:
            print("Error: SAM predictor not loaded. Cannot perform segmentation.")
            return [], display_image_resized
        sam_predictor_instance.set_image(raw_image)
        for bbox in bboxes_np:
            masks_sam, scores, _ = sam_predictor_instance.predict(box=bbox, multimask_output=True)
            seg_masks.append(masks_sam[np.argmax(scores)].astype(np.uint8))  # binary mask
    else:  # Use bounding boxes directly as masks
        for bbox in bboxes_np:
            zero_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            zero_mask[y1:y2, x1:x2] = 1
            seg_masks.append(zero_mask)

    segmented_display_image = display_image_resized.copy()
    if len(seg_masks) > 0:
        region_labels = [f"Region {i}" for i in range(len(seg_masks))]
        # draw_mask_and_number_on_image expects masks relative to the image they are drawn on.
        # It will resize masks if their dimensions don't match the display image.
        segmented_display_image = draw_mask_and_number_on_image(
            display_image_resized.copy(), seg_masks, region_labels,
            label_mode="1", alpha=0.5, anno_mode=["Mask", "Mark", "Box"]
        )
    return seg_masks, segmented_display_image


def get_depth_map(raw_image, depth_model_instance, depth_transform_instance, device_str):
    """ Generates a colorized depth map from the raw image. """
    if depth_model_instance is None or depth_transform_instance is None:
        print("Error: Depth model/transform not loaded. Cannot generate depth map.")
        return np.zeros_like(raw_image, dtype=np.uint8)

    orig_h, orig_w = raw_image.shape[:2]

    depth_input_image_dict = depth_transform_instance({"image": raw_image / 255.0})  # transform expects dict
    depth_input_image = depth_input_image_dict["image"]
    depth_input_image = torch.from_numpy(depth_input_image).unsqueeze(0).to(device_str)

    with torch.no_grad():
        raw_depth = depth_model_instance(depth_input_image)

    raw_depth = F.interpolate(raw_depth[None], (orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0]
    raw_depth = raw_depth.cpu().numpy()

    min_val, max_val = raw_depth.min(), raw_depth.max()
    if max_val - min_val > 1e-6:
        raw_depth = (raw_depth - min_val) / (max_val - min_val) * 255.0
    else:
        raw_depth = np.zeros_like(raw_depth)
    raw_depth = raw_depth.astype(np.uint8)
    colorized_depth = cv2.cvtColor(raw_depth, cv2.COLOR_GRAY2RGB)  # 3-channel
    return colorized_depth


def inference_vlm(
        input_str, raw_image, seg_masks_binary, colorized_depth_map,
        use_depth_flag, use_bfloat_flag, follow_up_flag,
        temperature_val, max_new_tokens_val, conv_mode_str,
        current_conv, current_conv_history,
        tokenizer_instance, spatialrgpt_model_instance, image_processor_instance, device_str
):
    if use_depth_flag:
        query_base = re.sub(r"<region\d+>", "<mask> <depth>", input_str)
    else:
        query_base = re.sub(r"<region\d+>", "<mask>", input_str)

    if not follow_up_flag or current_conv is None:
        current_conv = conv_templates[conv_mode_str].copy()
        if not follow_up_flag:  # New conversation explicitly started
            current_conv_history = {"user": [], "model": []}
            print("Starting a new conversation.")
        query = DEFAULT_IMAGE_TOKEN + "\n" + query_base
    else:  # Is a follow-up
        query = query_base

    print("Input query for VLM:", query)
    current_conv_history["user"].append(input_str)  # Store original user input

    current_conv.append_message(current_conv.roles[0], query)
    current_conv.append_message(current_conv.roles[1], None)
    prompt = current_conv.get_prompt()

    # Prepare image/depth tensors
    pil_raw_image = Image.fromarray(raw_image)
    pil_colorized_depth = Image.fromarray(colorized_depth_map)

    selected_dtype = torch.float16
    if use_bfloat_flag and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        selected_dtype = torch.bfloat16
    elif use_bfloat_flag and device_str == 'cuda':  # Check if it's CUDA for bfloat warning
        print("Warning: bfloat16 selected but not supported on this GPU. Using float16.")

    original_model_dtype = next(spatialrgpt_model_instance.parameters()).dtype
    if original_model_dtype != selected_dtype and device_str != 'cpu':
        spatialrgpt_model_instance.to(dtype=selected_dtype)

    images_tensor = process_images([pil_raw_image], image_processor_instance, spatialrgpt_model_instance.config).to(
        device_str, dtype=selected_dtype if device_str != 'cpu' else torch.float32)
    depths_tensor = process_images([pil_colorized_depth], image_processor_instance,
                                   spatialrgpt_model_instance.config).to(device_str,
                                                                         dtype=selected_dtype if device_str != 'cpu' else torch.float32)

    # Mask processing based on *current* input_str's regions
    current_input_region_tags = re.findall(r"<region(\d+)>", input_str)
    current_input_region_indices_int = [int(tag) for tag in current_input_region_tags]

    final_masks_for_model = None
    if len(seg_masks_binary) > 0 and len(current_input_region_indices_int) > 0:
        np_masks_for_processing = [np.array(Image.fromarray(m * 255)) for m in seg_masks_binary]

        _masks_tensor_all_available = process_regions(
            np_masks_for_processing,
            image_processor_instance,
            spatialrgpt_model_instance.config
        ).to(device_str, dtype=selected_dtype if device_str != 'cpu' else torch.float32)

        actual_mask_indices_to_pass_to_model = []
        for r_idx in current_input_region_indices_int:
            if 0 <= r_idx < _masks_tensor_all_available.size(0):
                actual_mask_indices_to_pass_to_model.append(r_idx)
            else:
                print(
                    f"Warning: Region index {r_idx} in current prompt is out of bounds for available masks ({_masks_tensor_all_available.size(0)} total). It will be ignored.")

        if actual_mask_indices_to_pass_to_model:
            final_masks_for_model = _masks_tensor_all_available[actual_mask_indices_to_pass_to_model]

    input_ids = tokenizer_image_token(prompt, tokenizer_instance, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
        0).to(device_str)

    stop_str = current_conv.sep if current_conv.sep_style != SeparatorStyle.TWO else current_conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer_instance, input_ids)

    with torch.inference_mode():
        output_ids = spatialrgpt_model_instance.generate(
            input_ids,
            images=[images_tensor],
            depths=[depths_tensor],
            masks=[final_masks_for_model] if final_masks_for_model is not None else None,
            do_sample=True if temperature_val > 0 else False,
            temperature=temperature_val,
            max_new_tokens=max_new_tokens_val,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    if original_model_dtype != selected_dtype and device_str != 'cpu':
        spatialrgpt_model_instance.to(dtype=original_model_dtype)

    outputs_raw = tokenizer_instance.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs_stripped = outputs_raw.strip()
    if outputs_stripped.endswith(stop_str):
        outputs_stripped = outputs_stripped[:-len(stop_str)]
    outputs_final_model = outputs_stripped.strip()

    print(f"Raw VLM output: {outputs_final_model}")

    mapping_dict = {str(out_idx): str(in_idx_tag) for out_idx, in_idx_tag in enumerate(current_input_region_tags)}
    remapped_outputs = outputs_final_model
    if mapping_dict:
        try:
            remapped_outputs = re.sub(r"\[([0-9]+)\]", lambda x: f"[{mapping_dict[x.group(1)]}]" if x.group(
                1) in mapping_dict else x.group(0), outputs_final_model)
        except KeyError as e:
            print(
                f"Output remapping failed: Model referred to index {e} not in current prompt's regions. Raw output shown.")

    print(f"Post-processed VLM output: {remapped_outputs}")

    if current_conv.messages and current_conv.messages[-1][0] == current_conv.roles[1] and current_conv.messages[-1][
        1] is None:
        current_conv.messages.pop()
    current_conv.append_message(current_conv.roles[1], outputs_raw)
    current_conv_history["model"].append(remapped_outputs)

    return remapped_outputs, current_conv, current_conv_history


def parse_box_string(box_str):
    try:
        coords = [int(coord) for coord in box_str.split(',')]
        if len(coords) != 4:
            raise ValueError("Box must have 4 coordinates.")
        x1, y1, x2, y2 = coords
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    except ValueError as e:
        print(f"Error: Invalid box format: {box_str}. {e}")
        print("Expected format: x1,y1,x2,y2 (all integers)")
        sys.exit(1)


def run_pipeline(
        parsed_args,
        sam_predictor_instance,
        depth_model_instance, depth_transform_instance,
        tokenizer_instance, spatialrgpt_model_instance, image_processor_instance,
        device_str,
        initial_conv_state, initial_conv_history_state
):
    # Load the image
    if not os.path.exists(parsed_args.image):
        print(f"Error: Image file not found: {parsed_args.image}")
        sys.exit(1)

    raw_image_bgr = cv2.imread(parsed_args.image)
    if raw_image_bgr is None:
        print(f"Error: Failed to load image: {parsed_args.image}")
        sys.exit(1)
    raw_image_rgb = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB)

    # Parse boxes
    boxes_xyxy = []
    if parsed_args.boxes:
        boxes_xyxy = [parse_box_string(box_str) for box_str in parsed_args.boxes]

    if not boxes_xyxy and re.search(r"<region\d+>", parsed_args.prompt):
        print("Warning: Prompt mentions regions (e.g., <region0>), but no boxes provided via --boxes.")
        print("Segmentation will not be performed for specific regions unless boxes are given.")

    print(f"Processing image: {parsed_args.image}")
    print(f"Using SAM for segmentation: {parsed_args.use_sam}")
    print(f"Using depth information: {parsed_args.use_depth}")

    seg_masks_binary, segmented_display_image = segment_using_boxes(
        raw_image_rgb, boxes_xyxy, parsed_args.use_sam, sam_predictor_instance
    )

    colorized_depth_map_for_vlm = np.zeros_like(raw_image_rgb, dtype=np.uint8)
    if parsed_args.use_depth:
        print("Generating depth map...")
        depth_map_output = get_depth_map(raw_image_rgb, depth_model_instance, depth_transform_instance, device_str)
        if depth_map_output is not None:
            colorized_depth_map_for_vlm = depth_map_output

    print(f"\nRunning inference with prompt: \"{parsed_args.prompt}\"")
    vlm_result, _, _ = inference_vlm(  # We don't need to use updated conv state in this single-run CLI
        parsed_args.prompt,
        raw_image_rgb,
        seg_masks_binary,
        colorized_depth_map_for_vlm,
        parsed_args.use_depth,
        parsed_args.use_bfloat,
        parsed_args.follow_up,
        parsed_args.temperature,
        parsed_args.max_new_tokens,
        parsed_args.conv_mode,
        initial_conv_state,
        initial_conv_history_state,
        tokenizer_instance,
        spatialrgpt_model_instance,
        image_processor_instance,
        device_str
    )

    if parsed_args.output:
        output_path = Path(parsed_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), cv2.cvtColor(segmented_display_image, cv2.COLOR_RGB2BGR))
        print(f"Segmented image saved to {output_path}")

        if parsed_args.use_depth and colorized_depth_map_for_vlm is not None and np.any(colorized_depth_map_for_vlm):
            depth_output_path = output_path.with_name(output_path.stem + "_depth" + output_path.suffix)
            cv2.imwrite(str(depth_output_path), cv2.cvtColor(colorized_depth_map_for_vlm, cv2.COLOR_RGB2BGR))
            print(f"Depth map saved to {depth_output_path}")

    print("\n===== FINAL VLM RESPONSE =====")
    print(vlm_result)
    print("==============================")


def main():
    parser = argparse.ArgumentParser(description="SpatialRGPT CLI Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--boxes", type=str, nargs='+',
                        help="List of boxes in format 'x1,y1,x2,y2'. Each box is a new region (region0, region1, ...).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt with <regionX> references (e.g., 'Describe <region0>')")
    parser.add_argument("--output", type=str,
                        help="Path to save the segmented image (optional). Depth map will be saved as <output_name>_depth.<ext> if --use-depth.")

    parser.add_argument('--use-sam', dest='use_sam', action='store_true',
                        help="Use SAM for segmentation (enabled by default)")
    parser.add_argument('--no-use-sam', dest='use_sam', action='store_false', help="Do NOT use SAM for segmentation")
    parser.set_defaults(use_sam=True)

    parser.add_argument('--use-depth', dest='use_depth', action='store_true',
                        help="Use depth information (enabled by default)")
    parser.add_argument('--no-use-depth', dest='use_depth', action='store_false', help="Do NOT use depth information")
    parser.set_defaults(use_depth=True)

    parser.add_argument('--use-bfloat', dest='use_bfloat', action='store_true',
                        help="Use bfloat16 precision if GPU supports (enabled by default)")
    parser.add_argument('--no-use-bfloat', dest='use_bfloat', action='store_false',
                        help="Do NOT use bfloat16 precision")
    parser.set_defaults(use_bfloat=True)

    parser.add_argument("--follow-up", action="store_true", default=False,
                        help="Treat this prompt as a follow-up to a previous one in the same session (maintains conversation history).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for text generation (default: 0.2)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens for generation (default: 512)")
    parser.add_argument("--conv-mode", type=str, default="llama_3",
                        help="Conversation mode (e.g., 'llama_3', 'vicuna_v1', 'llava_v1') (default: 'llama_3')")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the SpatialRGPT model directory or HuggingFace repo name")
    parser.add_argument("--model-name", type=str, default="SpatialRGPT-VILA1.5-8B",
                        help="Specific model name if path is a directory with multiple models (default: 'SpatialRGPT-VILA1.5-8B')")

    cli_args = parser.parse_args()

    # --- Environment Variable Checks ---
    depth_anything_path_env = os.environ.get("DEPTH_ANYTHING_PATH")
    sam_ckpt_path_env = os.environ.get("SAM_CKPT_PATH")

    if not depth_anything_path_env:
        print("Error: DEPTH_ANYTHING_PATH environment variable not set.")
        print("Please set it to point to the root of the DepthAnythingV2 repository.")
        sys.exit(1)
    if not os.path.isdir(depth_anything_path_env):
        print(f"Error: DEPTH_ANYTHING_PATH ({depth_anything_path_env}) is not a valid directory.")
        sys.exit(1)

    if not sam_ckpt_path_env:
        print("Error: SAM_CKPT_PATH environment variable not set.")
        print("Please set it to point to the SAM checkpoint file.")
        sys.exit(1)
    if not os.path.isfile(sam_ckpt_path_env):
        print(f"Error: SAM checkpoint not found at {sam_ckpt_path_env}")
        sys.exit(1)

    # Determine device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    # Initialize conversation state (passed to inference_vlm, which handles new/follow-up logic)
    initial_conv = None
    initial_conv_history = {"user": [], "model": []}

    # Load models
    print(f"Loading SpatialRGPT model from {cli_args.model_path} (name: {cli_args.model_name})...")
    try:
        # context_len_loaded is not used elsewhere in this CLI script
        tokenizer, model_loaded, image_processor_loaded, _ = load_pretrained_model(
            cli_args.model_path, cli_args.model_name, device_map=None
        )
        if model_loaded is None:
            raise ValueError("SpatialRGPT model loading failed.")
        spatialrgpt_model = model_loaded.to(device_str).eval()
        image_processor = image_processor_loaded
        print("SpatialRGPT model loaded successfully.")
    except Exception as e:
        print(f"Error loading SpatialRGPT model: {e}")
        sys.exit(1)

    print("Loading SAM model...")
    sam_predictor = get_sam_predictor(device_str, sam_ckpt_path_env)
    if sam_predictor is None:
        sys.exit(1)

    print("Loading Depth model...")
    depth_model, depth_transform = get_depth_predictor(
        device_str,
        depth_anything_path_env,
        MODEL_CONFIGS_DA_V2,
        DEPTH_ENCODER_TYPE
    )
    if depth_model is None or depth_transform is None:
        sys.exit(1)

    # Run main CLI logic
    run_pipeline(
        cli_args,
        sam_predictor,
        depth_model, depth_transform,
        tokenizer, spatialrgpt_model, image_processor,
        device_str,
        initial_conv, initial_conv_history
    )


if __name__ == "__main__":
    main()