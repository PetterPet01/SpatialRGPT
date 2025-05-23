import argparse
import glob
import os
import random
import time
import warnings
import re
import json

import cv2
import numpy as np
from mmengine import Config
from osdsynth.processor.captions import CaptionImage
from osdsynth.processor.pointcloud import PointCloudReconstruction
from osdsynth.processor.instruction import PromptGenerator

# from osdsynth.processor.filter import FilterImage
from osdsynth.processor.segment import SegmentImage
from osdsynth.utils.logger import SkipImageException, save_detection_list_to_json, setup_logger
from tqdm import tqdm

# Suppressing all warnings
warnings.filterwarnings("ignore")


def main(args):
    """Main function to control the flow of the program."""
    # Parse arguments
    cfg = Config.fromfile(args.config)
    exp_name = args.name if args.name else args.timestamp

    # Create log folder
    cfg.log_folder = os.path.join(args.log_dir, exp_name)
    os.makedirs(os.path.abspath(cfg.log_folder), exist_ok=True)

    # Create Wis3D folder
    cfg.vis = args.vis
    cfg.wis3d_folder = os.path.join(args.log_dir, "Wis3D")
    os.makedirs(os.path.abspath(cfg.wis3d_folder), exist_ok=True)

    # Init the logger and log some basic info
    cfg.log_file = os.path.join(cfg.log_folder, f"{exp_name}_{args.timestamp}.log")
    logger = setup_logger()  # cfg.log_file
    logger.info(f"Config:\n{cfg.pretty_text}")

    # Dump config to log
    cfg.dump(os.path.join(cfg.log_folder, os.path.basename(args.config)))

    # Create output folder
    cfg.exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(os.path.abspath(cfg.exp_dir), exist_ok=True)

    # Create folder for output json
    cfg.json_folder = os.path.join(cfg.exp_dir, "json")
    os.makedirs(os.path.abspath(cfg.json_folder), exist_ok=True)

    global_data = glob.glob(f"{args.input}/*.jpg") + glob.glob(f"{args.input}/*.png")
    device = "cuda"

    annotate(cfg, global_data, logger, device)


def annotate(cfg, global_data, logger, device):

    random.shuffle(global_data)

    segmenter = SegmentImage(cfg, logger, device)
    reconstructor = PointCloudReconstruction(cfg, logger, device)
    captioner = CaptionImage(cfg, logger, device)
    prompter = PromptGenerator(cfg, logger, device)

    for i, filepath in tqdm(enumerate(global_data), ncols=25):
        filename = filepath.split("/")[-1].split(".")[0]
        print(f"Processing file: {filename}")

        progress_file_path = os.path.join(cfg.log_folder, f"{filename}.progress")
        if os.path.exists(progress_file_path) and cfg.check_exist:
            continue

        image_bgr = cv2.imread(filepath)
        image_bgr = cv2.resize(image_bgr, (int(640 / (image_bgr.shape[0]) * (image_bgr.shape[1])), 640))

        try:

            # Run tagging model and get openworld detections
            vis_som, detection_list = segmenter.process(image_bgr)

            # Lift 2D to 3D, 3D bbox informations are included in detection_list
            detection_list = reconstructor.process(filename, image_bgr, detection_list)

            # Get LLaVA local caption for each region, however, currently just use a <region> placeholder
            detection_list = captioner.process_local_caption(detection_list)

            # Save detection list to json
            detection_list_path = os.path.join(cfg.json_folder, f"{filename}.json")
            save_detection_list_to_json(detection_list, detection_list_path)

            # Generate instructions (facts) based on templates
            facts = prompter.evaluate_predicates_on_pairs(detection_list)

            batched_llm_prompts = prepare_llm_prompts(facts, detection_list)

            llm_prompts_path = os.path.join(cfg.json_folder, f"{filename}_llm_prompts.json")
            with open(llm_prompts_path, "w") as f:
                json.dump(batched_llm_prompts, f, indent=2)

            for llm_prompt in batched_llm_prompts:
                print(f"{llm_prompt}")
                print("-----------------------")

        except SkipImageException as e:
            # Meet skip image condition
            logger.info(f"Skipping processing {filename}: {e}.")
            continue


def prepare_llm_prompts(facts, detection_list):
    region_to_tag_list = []
    batched_instructions = []
    for qa_idx, instruction in enumerate(facts):
        i_regions = re.findall(r"<region(\d+)>", instruction)
        region_to_tag = {int(region): detection_list[int(region)]["class_name"] for region in i_regions}
        region_to_tag_list.append(region_to_tag)

        object_reference = []
        for r_id, (region, tag) in enumerate(region_to_tag.items()):
            object_reference.append(f"<region{region}> {tag}")
        object_reference = ", ".join(object_reference)

        new_instruction = f"[Objets]: {object_reference}. [Description]: {instruction}"
        batched_instructions.append(new_instruction)

    return batched_instructions


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(description="Generate 3D SceneGraph for an image.")
    parser.add_argument("--config", default="configs/v2.py", help="Annotation config file path.")
    parser.add_argument(
        "--input",
        default="./demo_images",
        help="Path to input, can be json of folder of images.",
    )
    parser.add_argument("--output-dir", default="./demo_out", help="Path to save the scene-graph JSON files.")
    parser.add_argument("--name", required=False, default=None, help="Specify, otherwise use timestamp as nameing.")
    parser.add_argument("--log-dir", default="./demo_out/log", help="Path to save logs and visualization results.")
    parser.add_argument("--vis", required=False, default=True, help="Wis3D visualization for reconstruted pointclouds.")
    parser.add_argument("--overwrite", required=False, action="store_true", help="Overwrite previous.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.timestamp = timestamp
    main(args)
