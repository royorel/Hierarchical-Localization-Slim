import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import h5py
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2

from . import logger, matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval


roma_transforms = [v2.Compose([v2.Resize(560),
                               v2.PILToTensor(),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
                              ]),
                   v2.Compose([v2.Resize(864),
                               v2.PILToTensor(),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
                              ]),
                  ]

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    "aliked+roma": {
        "output": "matches-aliked-roma",
        "requires_images": True,
        "white_bg": True,
        "transforms": roma_transforms,
        "model": {
            "name": "roma",
            'discard_match_th': 0.4,
            'device': 'cuda',
        },
    },
    "NN-superpoint": {
        "output": "matches-NN-mutual-dist.7",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
        },
    },
    "NN-ratio": {
        "output": "matches-NN-mutual-ratio.8",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
        },
    },
    "NN-mutual": {
        "output": "matches-NN-mutual",
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
        },
    },
}



class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r, 
                 requires_images=False, im_root=None, transforms=None, white_bg=False):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r
        self.requires_images = requires_images
        self.white_bg = white_bg
        if self.requires_images:
            self.im_root = im_root
            globs = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]
            self.transforms = transforms
            
            logger.info(f"Matcher requires raw images.")
            paths = []
            for g in globs:
                paths += glob.glob((Path(self.im_root) / "**" / g).as_posix(), recursive=True)
            if len(paths) == 0:
                raise ValueError(f"Could not find any image in root: {self.im_root}.")
            paths = sorted(set(paths))
            self.names = {Path(im_path).name: im_path for im_path in paths}
            logger.info(f"Found {len(self.names)} images in root {self.im_root}.")

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            # some matchers might expect an image but only use its size
            if self.requires_images:
                im0 = Image.open(self.names[name0])
                if im0.mode == "RGBA" and self.white_bg:
                    im_data = np.array(im0)
                    im, mask = im_data[...,:3], im_data[...,3:] / im_data[...,3:].max()
                    im0 = Image.fromarray((mask * im + 255 * (1-mask)).astype(np.uint8))
                
                if self.transforms is not None:
                    im0_out = []
                    for transform in self.transforms:
                        im0_out.append(transform(im0))
                
                data["image0"] = im0_out
            else:
                data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            if self.requires_images:
                im1 = Image.open(self.names[name1]).convert("RGB")
                if im0.mode == "RGBA" and self.white_bg:
                    im_data = np.array(im1)
                    im, mask = im_data[...,:3], im_data[...,3:] / im_data[...,3:].max()
                    im1 = Image.fromarray((mask * im + 255 * (1-mask)).astype(np.uint8))
                
                if self.transforms is not None:
                    im1_out = []
                    for transform in self.transforms:
                        im1_out.append(transform(im1))
                
                data["image1"] = im1_out
            else:
                data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp

    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
    image_dir: Optional[Path] = None,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite, image_dir)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
    image_dir: Path = None,
) -> Path:
    logger.info(
        "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info("Skipping the matching.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref,
                                  requires_images=conf.get('requires_images', False),
                                  im_root=image_dir,
                                  transforms=conf.get('transforms', None),
                                  white_bg=conf.get("white_bg", False),)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = names_to_pair(*pairs[idx])   
        writer_queue.put((pair, pred))

    writer_queue.join()
    logger.info("Finished exporting matches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--features", type=str, default="feats-superpoint-n4096-r1024")
    parser.add_argument("--matches", type=Path)
    parser.add_argument(
        "--conf", type=str, default="superglue", choices=list(confs.keys())
    )
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
