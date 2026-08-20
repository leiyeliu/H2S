try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import copy
import itertools
import logging

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import add_maskformer2_config
from h2s import (
    AVISDatasetMapper,
    AVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    add_avism_config,
)
from h2s.data.datasets.builtin import register_all_avis


AVIS_DATASET_NAMES = ("avis_train", "avis_val", "avis_test")


def register_avis_datasets(dataset_root, audio_feature_dir="FEATAudios_sep"):
    """Register AVISeg from an explicit root for this process."""
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"AVISeg dataset root does not exist: {dataset_root}")

    if os.path.isabs(audio_feature_dir) or ".." in audio_feature_dir.split(os.sep):
        raise ValueError("--audio-feature-dir must be relative to each dataset split")

    for dataset_name in AVIS_DATASET_NAMES:
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)
    register_all_avis(dataset_root, audio_feature_dir=audio_feature_dir)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return AVISEvaluator(dataset_name, cfg, False, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AVISDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        mapper = AVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        if cfg["eval_only"]:
            from torch.amp import autocast
            logger = logging.getLogger(__name__)
            if isinstance(evaluators, DatasetEvaluator):
                evaluators = [evaluators]
            if evaluators is not None:
                assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                    len(cfg.DATASETS.TEST), len(evaluators)
                )

            results = OrderedDict()
            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                data_loader = cls.build_test_loader(cfg, dataset_name)
                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = cls.build_evaluator(cfg, dataset_name)
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                with autocast('cuda'):
                    results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i

                print("AP: {} || AP_s: {} || AP_m: {} || AP_l: {} || AR: {}".format(results_i['segm']['AP_all'],
                                                                                    results_i['segm']['AP_s'],
                                                                                    results_i['segm']['AP_m'],
                                                                                    results_i['segm']['AP_l'],
                                                                                    results_i['segm']['AR_all']))

                print("DetA: {} || DetRe: {} || DetPr: {}".format(results_i['segm']['DetA'],
                                                                  results_i['segm']['DetRe'],
                                                                  results_i['segm']['DetPr']))

                print("AssA: {} || AssRe: {} || AssPr: {}".format(results_i['segm']['AssA'],
                                                                  results_i['segm']['AssRe'],
                                                                  results_i['segm']['AssPr']))

                print("HOTA: {} || LocA: {} || DetA: {} || AssA: {}".format(results_i['segm']['HOTA'],
                                                                            results_i['segm']['LocA'],
                                                                            results_i['segm']['DetA'],
                                                                            results_i['segm']['AssA']))

                print("FSLAn_count: {} || FSLAn_all: {} || FSLAs_count: {} || FSLAs_all: {} || FSLAm_count: {} || FSLAm_all: {}".format(
                    results_i['segm']['FAn_count'],
                    results_i['segm']['FAn_all'],
                    results_i['segm']['FAs_count'],
                    results_i['segm']['FAs_all'],
                    results_i['segm']['FAm_count'],
                    results_i['segm']['FAm_all']))

                print("FSLA: {} || FSLAn: {} || FSLAs: {} || FSLAm: {}".format(results_i['segm']['FA'],
                                                                               results_i['segm']['FAn'],
                                                                               results_i['segm']['FAs'],
                                                                               results_i['segm']['FAm']))

            if len(results) == 1:
                results = list(results.values())[0]
            return results
        else:
            pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    dataset_root = getattr(args, "dataset_root", None)
    if dataset_root is not None:
        register_avis_datasets(
            dataset_root,
            getattr(args, "audio_feature_dir", "FEATAudios_sep"),
        )

    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_avism_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg["eval_only"] = args.eval_only
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="h2s")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def build_argument_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "AVISeg root containing train.json/val.json/test.json and split "
            "directories. Defaults to Detectron2's existing dataset registration."
        ),
    )
    parser.add_argument(
        "--audio-feature-dir",
        default="FEATAudios_sep",
        help=(
            "Separated VGGish feature directory relative to each split "
            "(default: FEATAudios_sep)."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
