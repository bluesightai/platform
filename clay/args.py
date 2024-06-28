import torch
import yaml
from box import Box
from pydantic_settings import BaseSettings


class Args(BaseSettings):
    """These values may be overriden by envs"""

    hub_repo_id: str = "made-with-clay/Clay"
    hub_filename: str = "clay-v1-base.ckpt"
    clay_metadata_path: str = "./metadata.yaml"
    platform: str = "sentinel-2-l2a"


args = Args()
metadata = Box(yaml.safe_load(open(args.clay_metadata_path)))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
