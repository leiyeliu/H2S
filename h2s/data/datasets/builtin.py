import os

from .avis import (
    register_avis_instances,
    _get_avis_instances_meta,
)


# ==== Predefined splits for AVIS ===========
_PREDEFINED_SPLITS_AVIS = {
    "avis_train": ("train", "train/JPEGImages", "train.json"),
    "avis_val": ("val", "val/JPEGImages", "val.json"),
    "avis_test": ("test", "test/JPEGImages", "test.json"),
}


def register_all_avis(root, audio_feature_dir=None):
    audio_feature_dir = audio_feature_dir or os.getenv(
        "H2S_AUDIO_FEATURE_DIR", "FEATAudios_sep"
    )
    for key, (split, image_root, json_file) in _PREDEFINED_SPLITS_AVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_avis_instances(
            key,
            _get_avis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            os.path.join(root, split, audio_feature_dir),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_avis(_root)
