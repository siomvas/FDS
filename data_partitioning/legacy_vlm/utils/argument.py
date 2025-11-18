import os
import shutil
import argparse
from pathlib import Path
#from dotenv import load_dotenv
from time import strftime, localtime
from utils.constants import TINY_IMAGENET, FOOD101, PASCALVOC, ADE20K
from utils.constants import INTERNVL3_38B, INTERNVL3_8B, LLAMA33_70B, LLAMA31_8B_INSTRUCT , MINICPM26, QWEN3_8B, QWEN3_30B


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Experiment arguments for ICTC")

# ---datadir
parser.add_argument("--dataset", type=str, default=TINY_IMAGENET, choices=[TINY_IMAGENET, FOOD101, PASCALVOC, ADE20K])
# ---parse initial arguments first
args, _ = parser.parse_known_args()

# ---output dir
parser.add_argument("--exp_name", type=str, default="test")

# ---*** criteria choices and set up (depends on dataset) ***------------------
if args.dataset == TINY_IMAGENET:
    criteria_choices = ["main_object", "size", "time", "color", "location"]
    # prompt: embedded as a criterion in all prompts
    # examplar: in step 2a, num_class: embedded in step 2b & 3 for the num of class for clustering
    criteria_config = {
        0: {"prompt": "Main Object", "examplar": "Tree", "num_classes": 7},
        1: {"prompt": "Size and Scale of the main object", "examplar": "Large", "num_classes": 4},
        2: {"prompt": "Time of day", "examplar": "Morning", "num_classes": 4},
        3: {"prompt": "Dominant Color", "examplar": "Green", "num_classes": 7},
        4: {"prompt": "Location of the main object", "examplar": "Field", "num_classes": 5},
    }
elif args.dataset == FOOD101:
    criteria_choices = ["main_object", "size", "color", "texture", "angle", "lighting"]
    criteria_config = {
        0: {"prompt": "Main Dish", "examplar": "Pancake", "num_classes": 7}, # rerun this
        1: {"prompt": "Size and Scale of the dish", "examplar": "Big", "num_classes": 8},
        2: {"prompt": "Dominant color of the dish", "examplar": "Golden-brown", "num_classes": 8},
        3: {"prompt": "Texture of the dish", "examplar": "Soft", "num_classes": 8}, #**
        4: {"prompt": "Camera Angle", "examplar": "Overhead", "num_classes": 5},
        5: {"prompt": "Lighting Intensity", "examplar": "Bright", "num_classes": 5},
    }
elif args.dataset == ADE20K:
    criteria_choices = ["main_object", "scene", "light", "color", "size"]
    criteria_config = {
        0: {"prompt": "Main Object", "examplar": "Tree", "num_classes": 8},
        1: {"prompt": "Location", "examplar": "Field", "num_classes": 4},
        2: {"prompt": "Lighting Intensity", "examplar": "Bright", "num_classes": 4},
        3: {"prompt": "Dominant Color", "examplar": "Green", "num_classes": 7},
        4: {"prompt": "Size of the main object", "examplar": "Large", "num_classes": 5},
    }
elif args.dataset == PASCALVOC:
    # TODO: CHANGE THIS
    criteria_choices = ["", ""]
else:
    criteria_choices = ["main_object"]

parser.add_argument("--criteria", type=str, default="main_object", choices=criteria_choices)

# ---model choices
# for llama
parser.add_argument("--llama_ver", type=str, default=LLAMA33_70B, choices = [LLAMA33_70B, LLAMA31_8B_INSTRUCT])
# for vlm
parser.add_argument("--vlm_model", type=str, default=QWEN3_30B, choices = [MINICPM26, INTERNVL3_8B, INTERNVL3_38B, QWEN3_8B, QWEN3_30B])

args = parser.parse_args()

# load_dotenv()
# args.home_path = os.getenv("HOME_PATH")

# if not args.home_path:
    #raise EnvironmentError("HOME_PATH not found in .env file. Please define it before running.")

# Set this to the local git dir
args.home_path = "/path/to/FDS/"

#------ Setting the params ------- #
args.general_data_dir = f"{args.home_path}/data_partitioning/legacy_vlm/data/"
args.specific_dataset_dir = args.general_data_dir + args.dataset
args.output_path = args.specific_dataset_dir + f"/output_{args.criteria}_{args.exp_name}"

# ------ Create output folder for each run ------
Path(args.output_path).mkdir(parents=True, exist_ok=True)
args.step1_result_path = f"{args.output_path}/step1_result.jsonl"
args.step2a_result_path = f"{args.output_path}/step2a_result.jsonl"
args.step2b_result_path = f"{args.output_path}/step2b_result.txt"
args.step3_result_path = f"{args.output_path}/step3_result.jsonl"

# ----- Prompt setting
# Direction for each prompt
args.prompt_path = args.specific_dataset_dir
args.step1_prompt_path = args.prompt_path + f"/step1_prompt" #+ args.criteria 
args.step1_prompt_path += ".txt"
args.step2a_prompt_path = args.prompt_path + f"/step2a_prompt" #+ args.criteria 
args.step2a_prompt_path += ".txt"
args.step2b_prompt_path = args.prompt_path + f"/step2b_prompt" #+ args.criteria 
args.step2b_prompt_path += ".txt"
args.step3_prompt_path = args.prompt_path + f"/step3_prompt" #+ args.criteria 
args.step3_prompt_path += ".txt"


# Copy the prompts for each experiment
# shutil.copy(args.step1_prompt_path, f"{args.output_path}/step1_prompt.txt")
# shutil.copy(args.step2a_prompt_path, f"{args.output_path}/step2a_prompt.txt")
# shutil.copy(args.step2b_prompt_path, f"{args.output_path}/step2b_prompt.txt")
# shutil.copy(args.step3_prompt_path, f"{args.output_path}/step3_prompt.txt")

# ----- set up criteria for parsing into prompt --
# find index of the chosen criteria
criteria_index = criteria_choices.index(args.criteria)
# get corresponding prompt label
config = criteria_config.get(criteria_index, None)
if config:
    args.prompt_label = config["prompt"]
    args.examplar = config["examplar"]
    args.num_classes = config["num_classes"]
else:
    args.prompt_label = None
    args.examplar = None
    args.num_classes = None

print(f"Selected criteria: {args.criteria} (index {criteria_index})")
print(f"Prompt label: {args.prompt_label}")
print(f"Examplar: {args.examplar}")
print(f"Num class: {args.num_classes}")

#--------------------------------------------------------------------------------------