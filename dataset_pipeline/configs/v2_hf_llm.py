# Class related params
class_set = "ram"
add_bg_classes = False
accumu_classes = False
exp_suffix = None
rm_bg_classes = True
add_classes = []
remove_classes = [
    "room", "kitchen", "office", "house", "home", "building", "corner", "shadow", "carpet", "photo", "sea", "shade", "stall", "space", "aquarium", "apartment", "image", "city", "blue", "skylight", "hallway", "bureau", "modern", "salon", "doorway", "wall lamp", "scene", "sun", "sky", "smile", "cloudy", "comfort", "white", "black", "red", "green", "yellow", "purple", "pink", "stand", "wear", "area", "shine", "lay", "walk", "lead", "bite", "sing",
]
bg_classes = ["wall", "floor", "ceiling"]
sam_variant = "sam-hq"
specified_tags = "None"
box_threshold = 0.25
text_threshold = 0.2
nms_threshold = 0.5
masking_option = "none"
mask_area_threshold = 100
mask_conf_threshold = 0.3
max_bbox_area_ratio = 0.90
skip_bg = False
min_points_threshold = 50
min_points_threshold_after_denoise = 25
downsample_voxel_size = 0.02
dbscan_remove_noise = True
dbscan_eps = 0.15
dbscan_min_points = 15
spatial_sim_type = "overlap"
log_dir = "./demo_output_generalized_hf/logs"
vis = False
use_clip = False
perspective_model_variant = "perspective_fields"

# LLM related (Hugging Face)
# llm_model_name_hf = "Qwen/Qwen2.5-1.5B" # Smaller model for quick testing, e.g., Qwen/Qwen2-0.5B-Instruct or Qwen/Qwen2-1.5B-Instruct
llm_model_name_hf = "Qwen/Qwen2.5-7B-Instruct" # Larger model, requires more resources & HF_TOKEN
llm_max_retries = 2
llm_temperature = 0.1
llm_max_new_tokens = 1024 * 10 # Reduced for smaller models, adjust as needed
