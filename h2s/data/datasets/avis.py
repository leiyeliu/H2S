import contextlib
import io
import logging
import os
from pathlib import Path

import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

from .avis_api.avos import AVOS


"""
This file contains functions to parse AVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_avis_json", "register_avis_instances"]


AVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "violin"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "guitar"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "cello"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "flute"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "piano"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ukulele"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "accordion"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "guzheng"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "clarinet"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "cat"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "car"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "saxophone"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "dog"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "lawn_mover"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "tuba"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "banjo"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "pipa"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "bassoon"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "airplane"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "tree_harvester"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "trumpet"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "lion"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "bass"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "erhu"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "horse"}]


def _get_avis_instances_meta():
    thing_ids = [k["id"] for k in AVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in AVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 26, len(thing_ids)
    # Mapping from the incontiguous AVIS category id to an id in [0, 25]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in AVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def load_avis_json(
    json_file,
    image_root,
    dataset_name=None,
    extra_annotation_keys=None,
    audio_feature_root=None,
    num_audio_sources=8,
):

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        avis_api = AVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(avis_api.getCatIds())
        cats = avis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(avis_api.vids.keys())
    vids = avis_api.loadVids(vid_ids)

    anns = [avis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(avis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in AVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs

        if audio_feature_root is None:
            audio_feature_root = os.path.join(
                str(Path(image_root).parent), "FEATAudios_sep"
            )

        relative_frame_path = Path(vid_dict["file_names"][0])
        if len(relative_frame_path.parts) < 2:
            raise ValueError(
                "AVIS frame paths must include a video directory, got "
                f"{vid_dict['file_names'][0]!r}."
            )
        video_name = relative_frame_path.parts[0]
        audio_feature_paths = [
            os.path.join(
                audio_feature_root,
                video_name,
                f"source_{source_idx:02d}.npy",
            )
            for source_idx in range(1, num_audio_sources + 1)
        ]
        missing_audio_features = [
            path for path in audio_feature_paths if not os.path.isfile(path)
        ]
        if missing_audio_features:
            missing_preview = "\n  ".join(missing_audio_features[:3])
            raise FileNotFoundError(
                f"Missing {len(missing_audio_features)} separated audio feature "
                f"file(s) for video {video_name!r} under {audio_feature_root!r}. "
                "Run tools/preprocess_audio.py or pass the matching "
                "--audio-feature-dir to train_net.py. Missing:\n  "
                f"{missing_preview}"
            )
        # Keep paths in the catalog and load only the selected frames in the mapper.
        record["audio_feature_paths"] = audio_feature_paths

        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_avis_instances(
    name, metadata, json_file, image_root, audio_feature_root=None
):
    """
    Register a dataset in AVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "avis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
        audio_feature_root (str or path-like): directory containing one folder per
            video and ``source_01.npy`` through ``source_08.npy`` in each folder.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name,
        lambda: load_avis_json(
            json_file,
            image_root,
            name,
            audio_feature_root=audio_feature_root,
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        audio_feature_root=audio_feature_root,
        evaluator_type="avis",
        **metadata,
    )
