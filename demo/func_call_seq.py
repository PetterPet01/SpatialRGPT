#!/usr/bin/env python3
import re
import json
import argparse
import sys
import os
from typing import Dict, List, Any, Tuple, Optional, Union, Literal
import subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch  # Added for device checking

# Import Pydantic from LangChain
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser

# Import CLI module
from cli import (
    inference_vlm, segment_using_boxes, get_depth_map,
    get_sam_predictor, get_depth_predictor, load_pretrained_model,
    MODEL_CONFIGS_DA_V2, DEPTH_ENCODER_TYPE  # Added constants
)


# Pydantic Models for Output Formatting (some will be reused, new ones for iteration)
class FunctionCall(BaseModel):
    """Standard function call format"""
    name: str = Field(..., description="Function name")
    arguments: List[str] = Field(..., description="Function arguments")


class StandardFunctionOutput(BaseModel):
    """Standard function call output format"""
    function_call: FunctionCall = Field(..., description="Function call details")


class CountFunctionOutput(BaseModel):
    """Count function output format"""
    count_area: str = Field(..., description="Area to count in (only one area at a time)")
    count_instances: List[str] = Field(..., description="List of regions/instances to count")

    @validator('count_instances')
    def validate_count_instances(cls, v):
        if not v:
            raise ValueError("count_instances cannot be empty")
        return v


# New Pydantic Models for Iterative Agent Interaction
class ProposedFunctionCall(BaseModel):
    """Represents a function call proposed by the agent at a specific step."""
    call_details: Union[StandardFunctionOutput, CountFunctionOutput]
    reasoning: str = Field(..., description="Reasoning for this specific function call.")
    step_number: int


class AgentFinalAnswer(BaseModel):
    """Represents the final answer compiled by the agent."""
    answer: str = Field(..., description="The consolidated final answer to the original question.")
    reasoning: str = Field(..., description="Summary of how the answer was derived.")
    supporting_evidence: List[Dict[str, Any]] = Field(..., description="List of function calls made and their results.")


class AgentTurn(BaseModel):
    """Output of the agent at each turn of the interaction."""
    action_type: Literal["propose_call", "final_answer_ready"] = Field(...,
                                                                       description="The type of action the agent is taking.")
    proposed_call: Optional[ProposedFunctionCall] = Field(None,
                                                          description="Details of the function call being proposed, if action_type is 'propose_call'.")
    final_answer_details: Optional[AgentFinalAnswer] = Field(None,
                                                             description="The final answer, if action_type is 'final_answer_ready'.")
    original_question: str  # Keep track of the original question throughout turns for context
    scene_understanding_summary: Optional[str] = None  # Provide initial scene understanding


class SpatialRGPTAgent:
    def __init__(self):
        """Initialize the SpatialRGPT Function Call Agent"""
        self.function_descriptions = self._load_function_descriptions()

        # State for a single task
        self.original_question: Optional[str] = None
        self.image_path: Optional[str] = None
        self.region_objects: Dict[str, Dict] = {}
        self.rle_masks: List[Dict] = []
        self.current_scene_understanding: Optional[str] = None
        self.current_plan: List[Union[StandardFunctionOutput, CountFunctionOutput]] = []
        self.current_plan_step_index: int = 0
        self.accumulated_step_results: List[Dict[str, Any]] = []  # Stores {"call_details": ..., "result": ...}

        # Model components (will be initialized when needed)
        self.tokenizer = None
        self.spatialrgpt_model = None
        self.image_processor = None
        self.sam_predictor = None
        self.depth_model = None
        self.depth_transform = None
        self.device_str: Optional[str] = None
        self.conv_mode: Optional[str] = None

    def _load_function_descriptions(self) -> List[Dict]:
        """Load function descriptions from the predefined list"""
        return [
            {
                "name": "left_predicate",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Determines if object A is to the left of object B. Use for questions like 'Is X left of Y?'"
            },
            {
                "name": "below_predicate",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Checks if object A is below object B. Use for questions like 'Is X below Y?'"
            },
            {
                "name": "short_predicate",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Compares the height of object A to object B. Use for questions like 'Is X shorter than Y?'"
            },
            {
                "name": "thin_predicate",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Determines if object A is thinner than object B. Use for questions like 'Is X thinner than Y?'"
            },
            {
                "name": "small_predicate",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Evaluates if object A is smaller than object B. Use for questions like 'Is X smaller than Y?'"
            },
            {
                "name": "front_predicate",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Checks if object A is in front of object B. Use for questions like 'Is X in front of Y?'"
            },
            {
                "name": "vertical_distance_data",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Calculates the vertical distance between objects A and B. Use for questions like 'What is the vertical distance between X and Y?'"
            },
            {
                "name": "distance",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Computes the 3D distance between objects A and B. Use for questions like 'How far is X from Y?' or 'What is the distance between X and Y?'"
            },
            {
                "name": "horizontal_distance_data",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Calculates the horizontal distance between objects A and B. Use for questions like 'What is the horizontal distance between X and Y?'"
            },
            {
                "name": "width_data",
                "number_args": "1",
                "args_names": ["object_id"],
                "description": "Measures the width of object A. Use for questions like 'What is the width of X?'"
            },
            {
                "name": "height_data",
                "number_args": "1",
                "args_names": ["object_id"],
                "description": "Measures the height of object A. Use for questions like 'What is the height of X?'"
            },
            {
                "name": "direction",
                "number_args": "2",
                "args_names": ["object_A", "object_B"],
                "description": "Determines the clock-wise direction of object A relative to object B. Use for questions like 'What direction is X from Y?'"
            },
            {
                "name": "count",
                "number_args": "1",
                "args_names": ["target_description"],
                "description": "Counts the number of objects matching a description or within a specified region. Use for questions like 'How many objects are in Region X?' or 'How many pallet zones are there?'"
            }
        ]

    def initialize_models(self, model_path: str = "a8cheng/SpatialRGPT-VILA1.5-8B",
                          model_name: str = "vila-siglip-llama-3b",
                          conv_mode: str = "llama_3"):
        if self.tokenizer is not None:  # Models already initialized
            return

        print("🔄 Initializing models...")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device_str}")
        self.conv_mode = conv_mode

        # --- Environment Variable Checks ---
        depth_anything_root_path = os.environ.get("DEPTH_ANYTHING_PATH")
        sam_checkpoint_path = os.environ.get("SAM_CKPT_PATH")

        if not depth_anything_root_path:
            raise EnvironmentError(
                "DEPTH_ANYTHING_PATH environment variable not set. Please set it to point to the root of the DepthAnythingV2 repository.")
        if not os.path.isdir(depth_anything_root_path):
            raise EnvironmentError(f"DEPTH_ANYTHING_PATH ({depth_anything_root_path}) is not a valid directory.")

        if not sam_checkpoint_path:
            raise EnvironmentError(
                "SAM_CKPT_PATH environment variable not set. Please set it to point to the SAM checkpoint file.")
        if not os.path.isfile(sam_checkpoint_path):
            raise EnvironmentError(f"SAM checkpoint not found at {sam_checkpoint_path}")

        print(f"Loading SpatialRGPT model from: {model_path} (name: {model_name})...")
        # context_len_loaded is not used elsewhere in this script, so _ is fine
        tokenizer_loaded, spatialrgpt_model_loaded, image_processor_loaded, _ = load_pretrained_model(
            model_path, model_name, device_map=None
        )
        if spatialrgpt_model_loaded is None:
            raise RuntimeError("Failed to load SpatialRGPT model.")
        self.spatialrgpt_model = spatialrgpt_model_loaded.to(self.device_str).eval()
        self.tokenizer = tokenizer_loaded
        self.image_processor = image_processor_loaded

        print("Loading SAM model...")
        self.sam_predictor = get_sam_predictor(self.device_str, sam_checkpoint_path)
        if self.sam_predictor is None:
            raise RuntimeError("Failed to load SAM model.")

        print("Loading Depth model...")
        self.depth_model, self.depth_transform = get_depth_predictor(
            self.device_str,
            depth_anything_root_path,
            MODEL_CONFIGS_DA_V2,
            DEPTH_ENCODER_TYPE
        )
        if self.depth_model is None or self.depth_transform is None:
            raise RuntimeError("Failed to load Depth model.")

        print("✅ All models initialized successfully!")

    def _decode_rle_mask(self, rle_dict: Dict) -> np.ndarray:
        from pycocotools import mask as mask_utils  # Local import for safety
        rle = {
            'size': rle_dict['size'],
            'counts': rle_dict['counts'].encode('utf-8') if isinstance(rle_dict['counts'], str) else rle_dict['counts']
        }
        return mask_utils.decode(rle)

    def _convert_rle_masks_to_segmentation(self, image_rgb: np.ndarray) -> List[np.ndarray]:  # Takes RGB image
        seg_masks = []
        for rle_dict in self.rle_masks:
            try:
                binary_mask = self._decode_rle_mask(rle_dict)
                if binary_mask.shape[:2] != image_rgb.shape[:2]:
                    binary_mask = cv2.resize(binary_mask.astype(np.uint8),
                                             (image_rgb.shape[1], image_rgb.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                seg_masks.append(binary_mask.astype(np.uint8))  # Ensure binary 0/1
            except Exception as e:
                print(f"Warning: Failed to decode RLE mask: {e}")
                seg_masks.append(np.zeros(image_rgb.shape[:2], dtype=np.uint8))
        return seg_masks

    def _convert_masks_to_regions(self, query: str) -> str:
        # (Content of _convert_masks_to_regions remains the same)
        region_counter = 0
        converted_query = query

        while '<mask>' in converted_query:
            converted_query = converted_query.replace('<mask>', f'<region{region_counter}>', 1)
            region_counter += 1

        mask_pattern = r'<mask_(\d+)>'
        matches = re.findall(mask_pattern, converted_query)
        for match in matches:
            converted_query = re.sub(f'<mask_{match}>', f'<region{match}>', converted_query)

        return converted_query

    def _run_spatial_inference_with_rle(self, image_path: str, prompt: str,
                                        use_depth: bool = True, **kwargs) -> str:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None: raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        seg_masks_binary = self._convert_rle_masks_to_segmentation(image_rgb)

        colorized_depth_map_for_vlm = np.zeros_like(image_rgb, dtype=np.uint8)
        if use_depth:
            if self.depth_model and self.depth_transform:
                colorized_depth_map_for_vlm = get_depth_map(
                    image_rgb,
                    self.depth_model,
                    self.depth_transform,
                    self.device_str
                )
            else:
                print("Warning: Depth requested but depth model/transform not available. Skipping depth.")

        # Call the refactored cli.inference_vlm
        vlm_result_text, _updated_conv, _updated_conv_history = inference_vlm(
            input_str=prompt,
            raw_image=image_rgb,
            seg_masks_binary=seg_masks_binary,
            colorized_depth_map=colorized_depth_map_for_vlm,
            use_depth_flag=use_depth,
            use_bfloat_flag=kwargs.get('use_bfloat', True),
            follow_up_flag=False,
            temperature_val=kwargs.get('temperature', 0.2),
            max_new_tokens_val=kwargs.get('max_new_tokens', 512),
            conv_mode_str=self.conv_mode,
            current_conv=None,
            current_conv_history={"user": [], "model": []},
            tokenizer_instance=self.tokenizer,
            spatialrgpt_model_instance=self.spatialrgpt_model,
            image_processor_instance=self.image_processor,
            device_str=self.device_str
        )
        return vlm_result_text

    def _understand_scene(self, image_path: str) -> str:
        understanding_prompt = """Describe this warehouse scene. Identify all the objects in each region:
- What types of objects are visible (pallets, transporters, etc.)?
- What are their colors and positions?
- Which regions contain pallets vs transporters?
- Describe the spatial layout and relationships between objects.

Please be specific about each region and what it contains."""

        region_prompt = self._convert_masks_to_regions(understanding_prompt)

        return self._run_spatial_inference_with_rle(
            image_path=image_path,
            prompt=region_prompt,
            use_depth=True,
            temperature=0.1,
            max_new_tokens=1024
        )

    def _analyze_question(self, question: str) -> Dict[str, Any]:
        # (Content of _analyze_question remains the same)
        analysis = {
            "question_type": "complex",
            "required_functions": [],
            "reasoning_steps": [],
            "objects_of_interest": []
        }

        if "transporter" in question.lower(): analysis["objects_of_interest"].append("transporter")
        if "pallet" in question.lower(): analysis["objects_of_interest"].append("pallet")

        if "best choice" in question.lower() or "optimal" in question.lower():
            analysis["question_type"] = "optimization"
            analysis["reasoning_steps"] = [
                "Identify all available transporters and pallets.",
                "Calculate distances between relevant transporters and pallets.",
                "Check spatial relationships (e.g., occlusion) for accessibility.",
                "Determine optimal pallet based on proximity and constraints."
            ]
            analysis["required_functions"] = ["distance", "direction", "front_predicate", "count"]

        elif "distance" in question.lower():
            analysis["required_functions"] = ["distance", "horizontal_distance_data", "vertical_distance_data"]
            analysis["reasoning_steps"] = ["Identify the two objects/regions.",
                                           "Calculate the specified distance between them."]

        elif "count" in question.lower() or "how many" in question.lower():
            analysis["required_functions"] = ["count"]
            analysis["reasoning_steps"] = ["Identify what needs to be counted and in what area."]

        elif any(word in question.lower() for word in ["left", "right", "above", "below", "front", "behind"]):
            analysis["required_functions"] = ["left_predicate", "below_predicate", "front_predicate", "direction"]
            analysis["reasoning_steps"] = ["Identify the two objects/regions.",
                                           "Determine the specified spatial relationship."]
        else:
            analysis["reasoning_steps"] = ["Understand the query and determine relevant information needed."]

        return analysis

    def _create_standard_function_output(self, name: str, arguments: List[str]) -> StandardFunctionOutput:
        return StandardFunctionOutput(function_call=FunctionCall(name=name, arguments=arguments))

    def _create_count_function_output(self, count_area: str, count_instances: List[str]) -> CountFunctionOutput:
        return CountFunctionOutput(count_area=count_area, count_instances=count_instances)

    def _generate_optimization_calls_for_plan(self) -> List[Union[StandardFunctionOutput, CountFunctionOutput]]:
        calls = []
        transporters = [rid for rid, obj in self.region_objects.items() if "transporter" in obj["label"].lower()]
        pallets = [rid for rid, obj in self.region_objects.items() if "pallet" in obj["label"].lower()]

        if not transporters: calls.append(self._create_count_function_output("entire warehouse", ["transporter"]))
        if not pallets: calls.append(self._create_count_function_output("entire warehouse", ["pallet"]))

        for t in transporters:
            for p in pallets:
                calls.append(self._create_standard_function_output("distance", [t, p]))
        for i, p1 in enumerate(pallets):
            for p2 in pallets[i + 1:]:
                calls.append(self._create_standard_function_output("front_predicate", [p1, p2]))
                calls.append(self._create_standard_function_output("front_predicate", [p2, p1]))
        return calls

    def _generate_count_calls_for_plan(self, question: str) -> List[CountFunctionOutput]:
        calls = []
        if "pallet" in question.lower():
            calls.append(self._create_count_function_output("entire warehouse", ["pallet"]))
        if "transporter" in question.lower():
            calls.append(self._create_count_function_output("entire warehouse", ["transporter"]))

        region_matches = re.findall(r'<region(\d+)>', question)
        if not calls and region_matches:
            for region_id_str in region_matches:
                region_key = f"region{region_id_str}"
                if region_key in self.region_objects:
                    calls.append(
                        self._create_count_function_output(region_key, [self.region_objects[region_key]["label"]]))
                else:
                    calls.append(self._create_count_function_output(region_key, ["any object"]))

        if not calls:
            calls.append(self._create_count_function_output("entire warehouse", ["any object"]))
        return calls

    def _generate_spatial_relation_calls_for_plan(self, question: str) -> List[StandardFunctionOutput]:
        calls = []
        region_matches = re.findall(r'<region(\d+)>', question)
        if len(region_matches) >= 2:
            r_a, r_b = f"region{region_matches[0]}", f"region{region_matches[1]}"
            if "left" in question.lower(): calls.append(
                self._create_standard_function_output("left_predicate", [r_a, r_b]))
            if "below" in question.lower(): calls.append(
                self._create_standard_function_output("below_predicate", [r_a, r_b]))
            if "front" in question.lower(): calls.append(
                self._create_standard_function_output("front_predicate", [r_a, r_b]))
        return calls

    def _generate_distance_calls_for_plan(self, question: str) -> List[StandardFunctionOutput]:
        calls = []
        region_matches = re.findall(r'<region(\d+)>', question)
        if len(region_matches) >= 2:
            r_a, r_b = f"region{region_matches[0]}", f"region{region_matches[1]}"
            if "horizontal distance" in question.lower():
                calls.append(self._create_standard_function_output("horizontal_distance_data", [r_a, r_b]))
            elif "vertical distance" in question.lower():
                calls.append(self._create_standard_function_output("vertical_distance_data", [r_a, r_b]))
            else:
                calls.append(self._create_standard_function_output("distance", [r_a, r_b]))
        return calls

    def _create_function_call_plan(self, question: str, scene_understanding: str) -> List[
        Union[StandardFunctionOutput, CountFunctionOutput]]:
        print("📝 Creating function call plan...")
        analysis = self._analyze_question(question)
        converted_question = self._convert_masks_to_regions(question)
        plan = []
        if analysis["question_type"] == "optimization":
            plan.extend(self._generate_optimization_calls_for_plan())
        elif "count" in analysis["required_functions"]:
            plan.extend(self._generate_count_calls_for_plan(converted_question))
        if any(p in analysis["required_functions"] for p in ["left_predicate", "below_predicate", "front_predicate"]):
            plan.extend(self._generate_spatial_relation_calls_for_plan(converted_question))
        if any(d in analysis["required_functions"] for d in
               ["distance", "horizontal_distance_data", "vertical_distance_data"]):
            plan.extend(self._generate_distance_calls_for_plan(converted_question))

        if not plan:
            print("Warning: No specific function calls planned. This might indicate an unhandled question type.")
        print(f"📋 Plan created with {len(plan)} function call(s).")
        return plan

    def _generate_reasoning_for_call(self, func_output: Union[StandardFunctionOutput, CountFunctionOutput]) -> str:
        if isinstance(func_output, StandardFunctionOutput):
            name = func_output.function_call.name
            args = func_output.function_call.arguments
            if name == "distance": return f"To determine proximity, calculating distance between {args[0]} and {args[1]}."
            if "predicate" in name: return f"To understand spatial layout, checking relationship '{name}' between {args[0]} and {args[1]}."
            if "data" in name: return f"To get specific measurement, retrieving '{name}' for {', '.join(args)}."
            return f"Executing function '{name}' with arguments {args} as part of the plan."
        elif isinstance(func_output, CountFunctionOutput):
            return f"To quantify objects, counting '{', '.join(func_output.count_instances)}' in area '{func_output.count_area}'."
        return "Executing planned function call."

    def _get_next_agent_action(self) -> AgentTurn:
        if self.current_plan_step_index < len(self.current_plan):
            call_to_propose = self.current_plan[self.current_plan_step_index]
            reasoning = self._generate_reasoning_for_call(call_to_propose)
            proposed_call = ProposedFunctionCall(
                call_details=call_to_propose,
                reasoning=reasoning,
                step_number=self.current_plan_step_index + 1
            )
            return AgentTurn(
                action_type="propose_call",
                proposed_call=proposed_call,
                original_question=self.original_question,
                scene_understanding_summary=self.current_scene_understanding if self.current_plan_step_index == 0 else None
            )
        else:
            print("✅ Plan complete. Synthesizing final answer...")
            final_answer_text = self._synthesize_final_answer_from_history()
            final_answer_details = AgentFinalAnswer(
                answer=final_answer_text,
                reasoning="All planned steps have been executed and results compiled.",
                supporting_evidence=self.accumulated_step_results
            )
            return AgentTurn(
                action_type="final_answer_ready",
                final_answer_details=final_answer_details,
                original_question=self.original_question,
                scene_understanding_summary=self.current_scene_understanding
            )

    def _synthesize_final_answer_from_history(self) -> str:
        if not self.accumulated_step_results and not self.current_plan:
            if "describe" in self.original_question.lower() or "what do you see" in self.original_question.lower():
                return f"Based on the scene understanding: {self.current_scene_understanding}"
            return "I couldn't determine a specific course of action or gather information to answer the question with the planned steps."

        prompt_parts = [
            f"Original question: \"{self.original_question}\"",
            f"Scene understanding: \"{self.current_scene_understanding}\"",
            "The following information was gathered through function calls:"
        ]
        for i, step_result in enumerate(self.accumulated_step_results):
            call_details = step_result['call_details']  # This is now a dict
            result = step_result['result']
            # Reconstruct Pydantic model if needed for easy access, or access dict keys
            if call_details.get("function_call"):  # StandardFunctionOutput
                prompt_parts.append(
                    f"- Step {i + 1}: Called {call_details['function_call']['name']}({', '.join(call_details['function_call']['arguments'])}), Result: {result}")
            elif call_details.get("count_area"):  # CountFunctionOutput
                prompt_parts.append(
                    f"- Step {i + 1}: Counted {', '.join(call_details['count_instances'])} in {call_details['count_area']}, Result: {result}")

        prompt_parts.append(
            "\nBased on all the above, please provide a concise and direct answer to the original question.")
        synthesis_prompt = "\n".join(prompt_parts)

        print(f"🧠 Synthesizing with prompt:\n{synthesis_prompt[:500]}...")

        if not self.spatialrgpt_model:
            return f"Synthesized answer based on gathered information (models not loaded): {self.accumulated_step_results}"

        final_answer = self._run_spatial_inference_with_rle(
            image_path=self.image_path,
            prompt=synthesis_prompt,
            use_depth=False,
            temperature=0.3,
            max_new_tokens=512
        )
        return final_answer

    def start_task(self, question: str, image_path: str, region_objects: Dict,
                   rle_data: List[Dict], model_path_config: str, model_name_config: str,
                   conv_mode_config: str) -> AgentTurn:
        print(f"🚀 Starting new task for question: \"{question}\"")
        self.initialize_models(model_path_config, model_name_config, conv_mode_config)

        self.original_question = question
        self.image_path = image_path
        self.region_objects = region_objects
        self.rle_masks = rle_data
        self.current_plan_step_index = 0
        self.accumulated_step_results = []

        print("🖼️  Understanding scene...")
        self.current_scene_understanding = self._understand_scene(self.image_path)
        print("Scene understanding completed.")

        self.current_plan = self._create_function_call_plan(self.original_question, self.current_scene_understanding)

        return self._get_next_agent_action()

    def report_function_call_result(self, executed_call_details: Union[StandardFunctionOutput, CountFunctionOutput],
                                    result: Any) -> AgentTurn:
        print(f"📊 Agent received result for step {self.current_plan_step_index + 1}: {result}")
        self.accumulated_step_results.append({
            "call_details": executed_call_details.dict(),
            "result": result
        })
        self.current_plan_step_index += 1
        return self._get_next_agent_action()


def main():
    parser = argparse.ArgumentParser(description="Iterative SpatialRGPT Function Call Agent")
    parser.add_argument("--question", type=str, required=True, help="Question to process")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SpatialRGPT model (or HF identifier)")
    parser.add_argument("--model-name", type=str, default="vila-siglip-llama-3b",
                        help="Model name identifier for LLaVA")  # Default might need update based on common SpatialRGPT model names
    parser.add_argument("--conv-mode", type=str, default="llama_3", help="Conversation mode for LLaVA")
    parser.add_argument("--output-json", type=str, help="Path to save JSON output trace of the interaction")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts between steps")
    parser.add_argument("--region-objects", type=str,
                        help="Path to JSON file containing region objects (label, boundingbox)")
    parser.add_argument("--rle-masks", type=str, required=True,
                        help="Path to JSON file containing RLE masks for segmentation")

    args = parser.parse_args()

    if not os.path.exists(args.image): raise FileNotFoundError(f"Image file not found: {args.image}")

    if args.region_objects and os.path.exists(args.region_objects):
        with open(args.region_objects, 'r') as f:
            region_objects = json.load(f)
        print(f"📂 Loaded region objects from: {args.region_objects}")
    else:
        region_objects = {f"region{i}": {"label": f"object_in_region_{i}"} for i in range(20)}
        print(f"🔧 Using placeholder region object labels as --region-objects file was not found or provided.")

    region_objects = {
        "region0": {"label": "pallet", "boundingbox": [300, 300, 90, 90]},
        "region1": {"label": "pallet", "boundingbox": [320, 320, 90, 90]},
        "region2": {"label": "pallet", "boundingbox": [500, 450, 110, 110]},
        "region3": {"label": "pallet", "boundingbox": [850, 600, 100, 100]},
        "region4": {"label": "pallet", "boundingbox": [950, 500, 120, 120]},
        "region5": {"label": "transporter", "boundingbox": [1060, 520, 130, 70]},
        "region6": {"label": "pallet", "boundingbox": [800, 620, 100, 100]},
        "region7": {"label": "pallet", "boundingbox": [870, 350, 90, 90]},
        "region8": {"label": "transporter", "boundingbox": [950, 320, 130, 70]}
    }

    if not os.path.exists(args.rle_masks): raise FileNotFoundError(f"RLE masks file not found: {args.rle_masks}")
    with open(args.rle_masks, 'r') as f:
        rle_data = json.load(f)
    print(f"🎭 Loaded RLE masks from: {args.rle_masks}")

    agent = SpatialRGPTAgent()
    interaction_trace = []

    try:
        current_turn = agent.start_task(
            question=args.question,
            image_path=args.image,
            region_objects=region_objects,
            rle_data=rle_data,
            model_path_config=args.model_path,
            model_name_config=args.model_name,
            conv_mode_config=args.conv_mode
        )
        interaction_trace.append({"agent_turn": current_turn.dict()})

        if current_turn.scene_understanding_summary:
            print("\nInitial Scene Understanding:")
            print(current_turn.scene_understanding_summary)

        while current_turn.action_type == "propose_call":
            proposed_call_action = current_turn.proposed_call
            print(f"\nAGENT ACTION (Step {proposed_call_action.step_number}): Propose Function Call")
            print(f"💭 Reasoning: {proposed_call_action.reasoning}")

            call_details_obj = proposed_call_action.call_details
            print(call_details_obj)
            if isinstance(call_details_obj, StandardFunctionOutput):
                func_name = call_details_obj.function_call.name
                func_args = call_details_obj.function_call.arguments
                print(f"🔧 Proposed Call: {func_name}({', '.join(func_args)})")
                mock_result = f"Mocked result for {func_name} with args {func_args}"
            elif isinstance(call_details_obj, CountFunctionOutput):
                count_area = call_details_obj.count_area
                count_instances = call_details_obj.count_instances
                print(f"🔧 Proposed Call: Count({', '.join(count_instances)}) in Area: {count_area}")
                mock_result = f"Mocked count: {np.random.randint(0, 10)} for {', '.join(count_instances)} in {count_area}"
            else:
                mock_result = "Error: Unknown call type"
                print(f"Error: Unknown call type in proposal: {call_details_obj}")

            print(f"⚙️ PROCESSOR: Executing call... Result: \"{mock_result}\"")
            interaction_trace[-1]["processor_action"] = {
                "executed_call": call_details_obj.dict(),
                "mock_result": mock_result
            }

            if not args.non_interactive and agent.current_plan_step_index < len(agent.current_plan) - 1:
                input("Press Enter to continue to next agent step...")

            current_turn = agent.report_function_call_result(call_details_obj, mock_result)
            interaction_trace.append({"agent_turn": current_turn.dict()})

        if current_turn.action_type == "final_answer_ready":
            final_answer_action = current_turn.final_answer_details
            print("\nAGENT ACTION: Final Answer Ready")
            print(f"💭 Reasoning: {final_answer_action.reasoning}")
            print(f"✅ Final Answer: {final_answer_action.answer}")

        print("\n✅ Iterative processing completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        interaction_trace.append({"error": str(e), "traceback": traceback.format_exc()})

    finally:
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(interaction_trace, f, indent=2)
            print(f"\n💾 Interaction trace saved to: {args.output_json}")


if __name__ == "__main__":
    main()