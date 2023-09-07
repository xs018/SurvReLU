import time
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

import utils
from batchgenerators.utilities.file_and_folder_operations import *
from dataset.dataset_dino_torch import BrainTumorDataset
from dataset.aug import DataAugmentationDINO
from models.vit import VisionTransformer, vit_tiny
from models.vit_dino import DINOHead, MultiCropWrapper, DINOLoss


def save_hyperparams(args, save_dir):
    option = vars(args) ## args is the argparsing
    file_name = os.path.join(save_dir, 'hyper.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')


class DINO():
    args:object
    def __init__(self, args):
        args = args
        self.writer = SummaryWriter()
        save_hyperparams(args, self.writer.log_dir)

        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")

        # create model and move it to GPU with id rank
        self.device_id = rank % torch.cuda.device_count()

        # Fix random seeds
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.benchmark = True

    def initialize(self):
        # ============ preparing dataset ... ============
        transform = DataAugmentationDINO(112, 48,
                                         args.global_crops_scale,
                                         args.local_crops_scale,
                                         args.local_crops_number,)
        dataset = BrainTumorDataset(args.data_path, fold='all', transform=transform, aug_seg=False)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                 sampler=sampler,
                                                 batch_size=args.batch_size_per_gpu,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=False)
        print(f"Data loaded: there are {len(dataset)} images.")
        self.data_loader = data_loader

        # ============ preparing model ... ============
        student = vit_tiny(in_chans=4, img_size=[112], patch_size=args.patch_size, drop_path_rate=args.drop_path_rate)  # stochastic depth)
        teacher = vit_tiny(in_chans=4, img_size=[112], patch_size=args.patch_size) 
        embed_dim = student.embed_dim

        # multi-crop wrapper handles forward with inputs of different resolutions
        student = MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ))
        teacher = MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )

        # move networks to gpu
        student, teacher = student.to(self.device_id), teacher.to(self.device_id)

        # synchronize batch norms (if any)
        if utils.has_batchnorms(student):
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

            # we need DDP wrapper to have synchro batch norms working...
            teacher = DDP(teacher, device_ids=[self.device_id])
            teacher_without_ddp = teacher.module
        else:
            # teacher_without_ddp and teacher are the same thing
            teacher_without_ddp = teacher

        student = DDP(student, device_ids=[self.device_id])
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(student.module.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {args.arch} network.")
        self.student = student
        self.teacher = teacher
        self.teacher_without_ddp = teacher_without_ddp

         # ============ preparing loss ... ============
        dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda()
        self.dino_loss = dino_loss

        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(student)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif args.optimizer == "lars":
            optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        self.optimizer = optimizer

        # for mixed precision training
        fp16_scaler = None
        if args.use_fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()
        self.fp16_scaler = fp16_scaler

        # ============ init schedulers ... ============
        lr_schedule = utils.cosine_scheduler(
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs,
        )
        wd_schedule = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            args.epochs, len(data_loader),
        )

        self.lr_schedule = lr_schedule
        self.wd_schedule = wd_schedule

        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                                args.epochs, len(data_loader))
        self.momentum_schedule = momentum_schedule
        print(f"Loss, optimizer and schedulers ready.")

    def train(self):
        self.initialize()
        model_checkpoints_folder = join(self.writer.log_dir, 'checkpoints')
        maybe_mkdir_p(model_checkpoints_folder)

        # ============ optionally resume training ... ============
        to_restore = {"epoch": 0}
        utils.restart_from_checkpoint(
            os.path.join(model_checkpoints_folder, "model.pth"),
            run_variables=to_restore,
            student=self.student,
            teacher=self.teacher,
            optimizer=self.optimizer,
            fp16_scaler=self.fp16_scaler,
            dino_loss=self.dino_loss,
        )
        start_epoch = to_restore["epoch"]
        # start_time = time.time()
        print("Starting DINO training !")
        for epoch in range(start_epoch, args.epochs):
            self.data_loader.sampler.set_epoch(epoch)

            # ============ training one epoch of DINO ... ============
            train_stats = self.train_one_epoch(self.student, self.teacher, self.teacher_without_ddp, self.dino_loss,
                                               self.data_loader, self.optimizer, self.lr_schedule, self.wd_schedule, 
                                               self.momentum_schedule, epoch, self.fp16_scaler)

            # ============ writing logs ... ============
            save_dict = {
                'student': self.student.state_dict(),
                'teacher': self.teacher.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': self.dino_loss.state_dict(),
            }

            if self.fp16_scaler is not None:
                save_dict['fp16_scaler'] = self.fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(model_checkpoints_folder, 'model.pth'))
            if args.saveckp_freq and epoch % args.saveckp_freq == 0:
                utils.save_on_master(save_dict, os.path.join(model_checkpoints_folder, f'model{epoch:04}.pth'))
            
            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #          'epoch': epoch}
            break

    def train_one_epoch(self, student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler):
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            # update weight decay and learning rate according to their schedule
            it = len(data_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # move images to gpu
            images = batch['data']
            images = [im.cuda(non_blocking=True) for im in images]

            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, epoch)
            
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
            # student update
            optimizer.zero_grad()
            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            self.writer.add_scalar('train_loss', loss.item())
            self.writer.add_scalar('cosine_lr_decay', optimizer.param_groups[0]["lr"])
            self.writer.add_scalar('wd', optimizer.param_groups[0]["lr"])

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

            break

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    from run_dino import get_args_parser
    args = get_args_parser()
    DINO(args).train()