# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
import logging
import numpy as np
import os
import random
import argparse #<-- 추가됨
import sys  # <-- 이 줄 추가
import time # <-- 이 줄 추가

import torch
import torch.distributed as dist
from omegaconf import OmegaConf #<-- 추가됨

from dinov3.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader, make_dataset
import dinov3.distributed as distributed
from dinov3.eval.segmentation.eval import evaluate_segmentation_model
from dinov3.eval.segmentation.loss import MultiSegmentationLoss
from dinov3.eval.segmentation.metrics import SEGMENTATION_METRICS
from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.segmentation.schedulers import build_scheduler
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms, make_segmentation_train_transforms
from dinov3.logging import MetricLogger, SmoothedValue

# DINOv3 유틸리티 임포트 <-- 추가됨
import dinov3.models as models

def set_random_seed(seed):
    """모든 랜덤 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config, output_dir: str, time_prefix: str) -> None:
    """로깅을 설정하는 함수"""
    level = logging.INFO
    if distributed.is_main_process():
        level = logging.DEBUG if hasattr(config, "logging") and config.logging.debug else logging.INFO
    
    logger = logging.getLogger("dinov3")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    # 포맷터 생성
    date_format = "%Y-%m-%d %H:%M:%S"
    log_format = f"%(asctime)s | {time_prefix} | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 생성 (메인 프로세스에서만)
    if distributed.is_main_process():
        file_handler = logging.FileHandler(os.path.join(output_dir, f"{time_prefix}.log"), mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # config 파일 저장
        OmegaConf.save(config=config, f=os.path.join(output_dir, f"{time_prefix}.yaml"))


logger = logging.getLogger("dinov3")


class InfiniteDataloader:
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self.data_iterator = iter(dataloader)
        self.sampler = dataloader.sampler
        if not hasattr(self.sampler, "epoch"):
            self.sampler.epoch = 0  # type: ignore

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.dataloader)

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            self.sampler.epoch += 1
            self.data_iterator = iter(self.dataloader)
            data = next(self.data_iterator)
        return data


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def validate(
    segmentation_model: torch.nn.Module,
    val_dataloader,
    device,
    autocast_dtype,
    eval_res,
    eval_stride,
    decoder_head_type,
    num_classes,
    global_step,
    metric_to_save,
    current_best_metric_to_save_value,
):
    new_metric_values_dict = evaluate_segmentation_model(
        segmentation_model,
        val_dataloader,
        device,
        eval_res,
        eval_stride,
        decoder_head_type,
        num_classes,
        autocast_dtype,
    )
    logger.info(f"Step {global_step}: {new_metric_values_dict}")
    # `segmentation_model` is a module list of [backbone, decoder]
    # Only put the head in train mode
    segmentation_model.module.segmentation_model[1].train()
    is_better = False
    if new_metric_values_dict[metric_to_save] > current_best_metric_to_save_value:
        is_better = True
    return is_better, new_metric_values_dict


def train_step(
    segmentation_model: torch.nn.Module,
    batch,
    device,
    scaler,
    optimizer,
    optimizer_gradient_clip,
    scheduler,
    criterion,
    model_dtype,
    global_step,
):
    # a) load batch
    batch_img, (_, gt) = batch
    batch_img = batch_img.to(device)  # B x C x h x w
    gt = gt.to(device)  # B x (num_classes if multilabel) x h x w
    optimizer.zero_grad(set_to_none=True)

    # b) forward pass
    with torch.autocast("cuda", dtype=model_dtype, enabled=True if model_dtype is not None else False):
        pred = segmentation_model(batch_img)  # B x num_classes x h x w
        gt = torch.squeeze(gt).long()  # Adapt gt dimension to enable loss calculation

    # c) compute loss
    if gt.shape[-2:] != pred.shape[-2:]:
        pred = torch.nn.functional.interpolate(input=pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    loss = criterion(pred, gt)

    # d) optimization
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.module.parameters(), optimizer_gradient_clip)
        optimizer.step()

    if global_step > 0:  # inheritance from old mmcv code
        scheduler.step()

    return loss


def train_segmentation(
    backbone,
    config,
):
    # config.model_dtype 문자열 ('FLOAT32', 'FLOAT16' 등)을 
    # 실제 torch.dtype 객체 또는 None으로 변환합니다.
    
    autocast_dtype = None # 기본값 (FLOAT32인 경우)
    if config.model_dtype == "FLOAT16":
        autocast_dtype = torch.float16
    elif config.model_dtype == "BFLOAT16":
        autocast_dtype = torch.bfloat16
    elif config.model_dtype != "FLOAT32":
        logger.warning(f"Unknown model_dtype '{config.model_dtype}'. Defaulting to None (no autocast).")
    assert config.decoder_head.type == "linear", "Only linear head is supported for training"

    # 1- load the segmentation decoder
    logger.info("Initializing the segmentation model")
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        "linear",
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=autocast_dtype,
    )
    global_device = distributed.get_rank()
    local_device = torch.cuda.current_device()
    segmentation_model = torch.nn.parallel.DistributedDataParallel(
        segmentation_model.to(local_device), device_ids=[local_device]
    )  # should be local rank
    model_parameters = filter(lambda p: p.requires_grad, segmentation_model.parameters())
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model_parameters)}")

    # 2- create data transforms + dataloaders
    train_transforms = make_segmentation_train_transforms(
        img_size=config.transforms.train.img_size,
        random_img_size_ratio_range=config.transforms.train.random_img_size_ratio_range,
        crop_size=config.transforms.train.crop_size,
        flip_prob=config.transforms.train.flip_prob,
        reduce_zero_label=config.eval.reduce_zero_label,
    )
    val_transforms = make_segmentation_eval_transforms(
        img_size=config.transforms.eval.img_size,
        inference_mode=config.eval.mode,
    )

    train_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.train}:root={config.datasets.root}",
            transforms=train_transforms,
        )
    )
    train_sampler_type = None
    if distributed.is_enabled():
        train_sampler_type = SamplerType.DISTRIBUTED
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=global_device, seed=config.seed + global_device
    )
    train_dataloader = InfiniteDataloader(
        make_data_loader(
            dataset=train_dataset,
            batch_size=config.bs,
            num_workers=config.num_workers,
            sampler_type=train_sampler_type,
            shuffle=True,
            persistent_workers=False,
            worker_init_fn=init_fn,
        )
    )

    val_dataset = DatasetWithEnumeratedTargets(
        make_dataset(
            dataset_str=f"{config.datasets.val}:root={config.datasets.root}",
            transforms=val_transforms,
        )
    )
    val_sampler_type = None
    if distributed.is_enabled():
        val_sampler_type = SamplerType.DISTRIBUTED
    val_dataloader = make_data_loader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        sampler_type=val_sampler_type,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )

    # 3- define and create scaler, optimizer, scheduler, loss
    scaler = None
    if autocast_dtype is not None:
        scaler = torch.amp.GradScaler("cuda")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, segmentation_model.parameters()),
                "lr": config.optimizer.lr,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "weight_decay": config.optimizer.weight_decay,
            }
        ]
    )
    scheduler = build_scheduler(
        config.scheduler.type,
        optimizer=optimizer,
        lr=config.optimizer.lr,
        total_iter=config.scheduler.total_iter,
        constructor_kwargs=config.scheduler.constructor_kwargs,
    )
    criterion = MultiSegmentationLoss(
        diceloss_weight=config.train.diceloss_weight, celoss_weight=config.train.celoss_weight
    )
    total_iter = config.scheduler.total_iter
    global_step = 0
    global_best_metric_values = {metric: 0.0 for metric in SEGMENTATION_METRICS}

    # 5- train the model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("loss", SmoothedValue(window_size=4, fmt="{value:.3f}"))
    for batch in metric_logger.log_every(
        train_dataloader,
        50,
        header="Train: ",
        start_iteration=global_step,
        n_iterations=total_iter,
    ):
        if global_step >= total_iter:
            break
        loss = train_step(
            segmentation_model,
            batch,
            local_device,
            scaler,
            optimizer,
            config.optimizer.gradient_clip,
            scheduler,
            criterion,
            autocast_dtype,
            global_step,
        )
        global_step += 1
        metric_logger.update(loss=loss)
        if global_step % config.eval.eval_interval == 0:
            dist.barrier()
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict

        # one last validation only if the number of total iterations is NOT divisible by eval interval:
        if total_iter % config.eval.eval_interval:
            is_better, best_metric_values_dict = validate(
                segmentation_model,
                val_dataloader,
                local_device,
                autocast_dtype,
                config.eval.crop_size,
                config.eval.stride,
                config.decoder_head.type,
                config.decoder_head.num_classes,
                global_step,
                config.metric_to_save,
                global_best_metric_values[config.metric_to_save],
            )
            if is_better:
                logger.info(f"New best metrics at Step {global_step}: {best_metric_values_dict}")
                global_best_metric_values = best_metric_values_dict
    logger.info("Training is done!")
    # segmentation_model is a module list of [backbone, decoder]
    # Only save the decoder head
    torch.save(
        {
            "model": {k: v for k, v in segmentation_model.module.state_dict().items() if "segmentation_model.1" in k},
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(config.output_dir, "model_final.pth"),
    )
    logger.info(f"Final best metrics: {global_best_metric_values}")
    return global_best_metric_values


# --- main 함수 시작 (추가된 부분) ---

def get_args_parser():
    """커맨드 라인 인자를 파싱하기 위한 함수"""
    parser = argparse.ArgumentParser("DINOv3 Segmentation Training", add_help=False)
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the training configuration file (YAML)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the DINOv3 model (e.g., dinov3_vitl16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the trained model and logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (e.g. 'optimizer.lr=1e-4')",
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # 1. 설정 파일 로드
    config = OmegaConf.load(args.config_file)

    # 2. (선택적) 커맨드 라인에서 config 옵션 덮어쓰기
    if args.opts:
        cli_conf = OmegaConf.from_dotlist(args.opts)
        config = OmegaConf.merge(config, cli_conf)

    # 3. 커맨드 라인 인자를 config 객체에 추가
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)
    config.seed = args.seed # config 파일에 seed가 없어도 여기서 설정됨

    # 4. 분산 학습 모드 초기화
    distributed.enable()

    # 5. 로깅 설정
    setup_logging(config=config, output_dir=config.output_dir, time_prefix="")

    # 6. 랜덤 시드 설정
    # 각 프로세스(GPU)가 다른 시드를 갖도록 rank를 더해줌
    set_random_seed(config.seed + distributed.get_rank())

    logger.info("Config:\n%s", OmegaConf.to_yaml(config))
    logger.info("Command line args:\n%s", args)

    # 7. DINOv3 백본 모델 로드 (torch.hub.load 사용 - 제공된 코드 참조)
    logger.info(f"Loading backbone model via torch.hub: {args.model}")
    
    # REPO_DIR은 이 스크립트가 실행되는 dinov3 루트 디렉토리입니다.
    # source='local'은 REPO_DIR의 hubconf.py를 사용하라는 의미입니다.
    repo_dir = '/home/jwmaeng/dinov3'
    
    # 제공된 코드를 기반으로 weights 파일 경로를 동적으로 구성합니다.
    # (예: args.model == 'dinov3_vit7b16' -> 'weights/dinov3_vit7b16_pretrain.pth')
    weights_path = f"weights/{args.model}_pretrain.pth"
    
    logger.info(f"Attempting to load weights from: {weights_path}")
    
    if not os.path.exists(weights_path):
        logger.warning(f"Weights file not found at {weights_path}!")
        logger.warning("Attempting to load model using default 'pretrained=True'...")
        # (참고: 이 경로는 hubconf.py의 기본값을 따릅니다.)
        backbone = torch.hub.load(
            repo_dir,
            args.model,
            source='local',
            pretrained=True 
        )
    else:
        # 제공된 데모 코드와 동일한 방식으로 로드
        backbone = torch.hub.load(
            repo_dir,
            args.model,
            source='local',
            weights=weights_path
        )
    
    # 백본은 고정 (학습 안 함)
    backbone.eval()
    logger.info("Backbone model loaded and set to eval mode.")

    # 8. 학습 함수 호출
    logger.info("Starting segmentation training...")
    train_segmentation(backbone, config)

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
# --- main 함수 끝 ---