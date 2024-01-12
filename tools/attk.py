# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import sys

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


import copy
import tqdm
import shutil
import tempfile
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as tf
from torchvision import utils

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)


    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--use_fp16", type=bool, default=False)
    parser.add_argument("--lift3d_loca", type=str, default='/home/leheng.li/sim_opt/lift3d_1214/')
    parser.add_argument("--obj_loca", type=str, default='/home/leheng.li/sim_opt/lift3d_1214/stylegan_gen_perc/sample_sem_optim/')
    parser.add_argument("--num_of_car", type=int, default=1)
    parser.add_argument("--num_of_ins", type=int, default=100)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args





sys.path.append('/home/leheng.li/sim_opt/lift3d_1214/')
from box_utils import gen_3dbox_rays
from utils import get_rays_p2, resample_rays, get_rays_box_sample


# from ..lift3d_1214.box_utils import gen_3dbox_rays



# Full volume renderer
class PatchGen(nn.Module):
    def __init__(self, args, ):
        super().__init__()
        self.args = args
        self.lc_list = []

        ins_sel_id = 235
        car_sel = 'audiallround07'

        lc_loca = args.obj_loca + car_sel + '/%06d_1500.pth' % ins_sel_id
        self.lc_list.append(torch.load(lc_loca, map_location='cpu'))

        # names = os.listdir(args.obj_loca)[:1]
        # for name in names:
        #     for ins_id in range(0, 500, 500):
        #         lc_file = os.path.join(args.obj_loca, name, '%06d_1500.pth' % ins_id)
        #         self.lc_list.append(torch.load(lc_file, map_location='cpu'))

        from networks import Lift3D as Model
        from options import BaseOptions
        opt = BaseOptions().parse()
        self.generator = Model(opt.rendering, opt.model.style_dim)

        ckpt_path = 'checkpoint/lift3d_1219_fewpoint_lesschann_pixlossnomask/models_0200000.pt'
        ckpt = torch.load(args.lift3d_loca + ckpt_path, map_location='cpu')
        self.generator.load_state_dict(ckpt["g"])
        self.generator.whether_train = False
        self.generator.require_grad = False
        self.generator.cutpaste = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.generator = MMDataParallel(self.generator, device_ids=[0])

    def forward(self, cam_info, ins_num=2):
        batch_num = cam_info[0].shape[0]

        box_tensor = torch.rand((batch_num, ins_num, 7)).to(self.device)

        # box_tensor[:, :, 0] = - 4
        # box_tensor[:, :, 1] = 1.5
        # box_tensor[:, :, 2] = 12
        # box_tensor[:, :, 3] = 1.72
        # box_tensor[:, :, 4] = 1.95
        # box_tensor[:, :, 5] = 4.61
        # box_tensor[:, :, 6] = - np.pi / 2

        box_tensor[:, :, 0] = (box_tensor[:, :, 0] - 0.5) * 10
        box_tensor[:, :, 1] = torch.clip(torch.randn_like(box_tensor[:, :, 1]) / 10, min = -0.2, max = 0.2) + 1.5
        box_tensor[:, :, 2] = box_tensor[:, :, 2] * 5 + 10

        size_length = torch.clip(torch.randn_like(box_tensor[:, :, 3]) / 10, min = -0.2, max = 0.2) + 1
        box_tensor[:, :, 3] = size_length * 1.72 + torch.randn_like(size_length) / 20
        box_tensor[:, :, 4] = size_length * 1.95 + torch.randn_like(size_length) / 20
        box_tensor[:, :, 5] = size_length * 4.61 + torch.randn_like(size_length) / 20


        rota_offset = torch.clip(torch.randn_like(box_tensor[:, :, 6]) * \
                                 np.pi/2 * 0.5, min = -np.pi / 2, max = np.pi / 2)
        if torch.rand(1) > 0.5:
            box_tensor[:, :, 6] = rota_offset + np.pi / 2
        else:
            box_tensor[:, :, 6] = rota_offset - np.pi / 2

        # import ipdb; ipdb.set_trace()
        

        all_patchs = []
        for img_num in range(batch_num):
            curr_intrin = cam_info[0][img_num]
            curr_imgsize = cam_info[1][img_num]
            curr_patchs = []

            rescale = 1
            scale_factor = 0.5
            raw_imgsize = copy.deepcopy(curr_imgsize)
            raw_intrin = copy.deepcopy(curr_intrin)
            if rescale:
                curr_imgsize = [raw_imgsize[0] * scale_factor, raw_imgsize[1] * scale_factor, 3]
                curr_imgsize = [int(x) for x in curr_imgsize]
                curr_intrin[:2, ...] = curr_intrin[:2, ...] * scale_factor

            for obj_num in range(box_tensor.shape[1]):
                curr_box = box_tensor[img_num][obj_num]


                rays_full_dict = get_rays_p2(curr_intrin, curr_imgsize[:2])
                rays_full_dict, _, _ = resample_rays(rays_full_dict, rays_num=-1)
                rays_box_dict = get_rays_box_sample(rays_full_dict, curr_box)

                if rays_box_dict['world_info'] is None:
                    continue

                lc_id = np.random.randint(0, len(self.lc_list))
                lc_template = self.lc_list[lc_id]
                
                rays_box_dict['lc_shape'] = [x.to('cuda') for x in lc_template['lc_shape']]
                rays_box_dict['lc_color'] = [x.to('cuda') for x in lc_template['lc_color']]

                out = self.generator(None, add_opts=rays_box_dict)

                intersection_map = rays_box_dict['intersection_map']
                height, width = curr_imgsize[:2]

                rgb_out = torch.zeros((1, height, width, 3)).to(self.device) - 1
                semantic_out = torch.zeros((1, height, width, 32)).to(self.device)
                weight_out = torch.zeros((1, height, width, 1)).to(self.device)
                uv_grid = torch.meshgrid(torch.linspace(0, height-1, height), torch.linspace(0, width-1, width))
                uv_grid = torch.cat([uv.unsqueeze(-1) for uv in uv_grid], -1)

                uv_grid = uv_grid.reshape((-1, 2))
                uv_box = uv_grid[intersection_map].to(self.device).long()[None, ...]

                # import ipdb; ipdb.set_trace()
                

                # splat rays
                rgb_out[:, uv_box[:, :, 0], uv_box[:, :, 1], :] = out['rgb_map']
                semantic_out[:, uv_box[:, :, 0], uv_box[:, :, 1], :] = out['semantic_map']
                rgb_out = rgb_out.permute(0,3,1,2)
                semantic_out = semantic_out.permute(0,3,1,2)
                semantic_bin = torch.argmax(semantic_out, 1)     

                weight = torch.sum(out['weight'], -2)
                weight_out[:, uv_box[:, :, 0], uv_box[:, :, 1], :] = weight
                weight_out = weight_out.permute(0,3,1,2)

                pred_img = (rgb_out + 1) / 2
                pred_img = pred_img[:, [2,1,0], ...] * 255  # rgb to bgr
                # utils.save_image(rgb_out, 'output_raw.jpg')

                pred_mask = weight_out > 0.5
                pred_mask = pred_mask * 1.
                # import ipdb; ipdb.set_trace()



                gt_curr_box = curr_box.clone()

                if rescale:
                    pred_img = F.interpolate(pred_img, size=(raw_imgsize[0], raw_imgsize[1]), mode='bilinear', align_corners=True)
                    pred_mask = F.interpolate(pred_mask, size=(raw_imgsize[0], raw_imgsize[1]), mode='bilinear', align_corners=True)

                # gt_box2d = masks_to_boxes(pred_mask.clone().squeeze(0))
   
                try:
                    gt_box2d = masks_to_boxes(pred_mask.clone().squeeze(0))
                except:
                    continue

                # print(gt_curr_box)
                gt_center_depth = None

                gt_bboxes_3d = torch.cat([gt_curr_box[:3], gt_curr_box[5:6], gt_curr_box[3:4], gt_curr_box[4:5], gt_curr_box[6:]])[None, ...]  # hwl to lhw
                # gt_bboxes_3d = torch.cat([gt_curr_box[:3], gt_curr_box[5:6], gt_curr_box[3:4], gt_curr_box[4:5], -gt_curr_box[6:]])[None, ...]  # hwl to lhw

                curr_patchs.append([pred_img, pred_mask, [gt_box2d, gt_center_depth, gt_bboxes_3d]])

                
            all_patchs.append(curr_patchs)


        return all_patchs





def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # init_dist('pytorch', backend= 'nccl')
        # import ipdb; ipdb.set_trace()

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=4, dist=distributed, shuffle=False)

    train_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=4, dist=distributed, shuffle=True)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)





    # build val dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build train dataloader
    dataset_train = build_dataset(cfg.data.train)
    data_loader_train = build_dataloader(dataset_train, **train_loader_cfg)



    # build the model and load checkpoint
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE


    model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)

    patch_net = PatchGen(args)

    patch_net = MMDistributedDataParallel(
                patch_net.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=True,
                )



    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    # time.sleep(2)  # This line can prevent deadlock problem in some cases.

    device = model.device

    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    mean, std = mean[None, ..., None, None], std[None, ..., None, None]

    train_ins_num = 3
    val_ins_num = 3



    optimizer = optim.Adam(patch_net.module.lc_list[0]['lc_color'], lr=1e-3)

    for curr_epoch in range(100):

        for i, data in enumerate(tqdm.tqdm(data_loader_train)):
            # if i > 400:
            #     break
            optimizer.zero_grad()

            img_normed = data['img_inputs'][0]
            # back norm
            img_edit = img_normed * std
            img_edit = img_edit + mean

            img_edit_list = []
            for curr_batch in range(len(data['gt_labels_3d'].data[0])):

                data['gt_labels_3d'].data[0][curr_batch] = \
                             torch.zeros_like(data['gt_labels_3d'].data[0][curr_batch]) - 1

                cam_para = data['img_inputs'][3][curr_batch]
                img_size = [(900, 1600, 3)] * cam_para.shape[0]
                cam_info = [cam_para, img_size]
                

                with torch.cuda.amp.autocast():
                    patch_add = patch_net(cam_info, train_ins_num)

                tmp_list = []
                for curr_view in range(len(patch_add)):
                    curr_img = img_edit[curr_batch, curr_view:curr_view+1]

                    for curr_advobj in range(len(patch_add[curr_view])):
                        adv_patch, adv_mask, adv_label = patch_add[curr_view][curr_advobj]

                        height, width = 396, 704
                        adv_patch = tf.resize(adv_patch, (height, width))
                        adv_patch = adv_patch[:, :, 140:396, :704].cpu()

                        adv_mask = tf.resize(adv_mask, (height, width))
                        adv_mask = adv_mask[:, :, 140:396, :704].cpu()

                        # (704, 396)
                        # (900, 1600, 3)
                        # (396, 704, 3) resize
                        # (256, 704, 3) crop

                        curr_img = curr_img * (1 - adv_mask) + adv_patch * adv_mask
                    tmp_list.append(curr_img)
                
                img_edit_list.append(torch.cat(tmp_list, 0)[None])

            img_edit = torch.cat(img_edit_list, 0)


            # utils.save_image(img_edit.reshape((-1, 3, 256, 704))[:, [2,1,0], ...] / 255, 'output.jpg')

            # import ipdb; ipdb.set_trace()
            

            img_input = img_edit - mean
            img_input = img_input / std
            data['img_inputs'][0] = img_input

            result = model(return_loss=True, **data)

            loss = sum([*result.values()])

            loss.backward()
            optimizer.step()
            

            # if rank == 0:
            #     batch_size = data['img_inputs'][0].shape[0]
            #     for _ in range(batch_size * world_size):
            #         prog_bar_train.update()

        torch.save(patch_net.module.lc_list[0], 'lc_folder/lc_%04d.pth' % curr_epoch)

        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))

        results = []

        for i, data in enumerate(data_loader):
            with torch.no_grad():

                img_list = []
                img_normed = data['img_inputs'][0][0]
                
                # back norm
                img_edit = img_normed.clone() * std
                img_edit = img_edit + mean

                img_edit_list = []
                for curr_batch in range(data['img_inputs'][0][0].shape[0]):

                    cam_para = data['img_inputs'][0][3][curr_batch]
                    img_size = [(900, 1600, 3)] * cam_para.shape[0]
                    cam_info = [cam_para, img_size]

                    with torch.cuda.amp.autocast():
                        patch_add = patch_net(cam_info, val_ins_num)

                    tmp_list = []
                    for curr_view in range(len(patch_add)):
                        curr_img = img_edit[curr_batch, curr_view:curr_view+1]

                        for curr_advobj in range(len(patch_add[curr_view])):
                            adv_patch, adv_mask, adv_label = patch_add[curr_view][curr_advobj]

                            height, width = 396, 704
                            adv_patch = tf.resize(adv_patch, (height, width))
                            adv_patch = adv_patch[:, :, 140:396, :704].cpu()

                            adv_mask = tf.resize(adv_mask, (height, width))
                            adv_mask = adv_mask[:, :, 140:396, :704].cpu()

                            curr_img = curr_img * (1 - adv_mask) + adv_patch * adv_mask
                        tmp_list.append(curr_img)
                    
                    img_edit_list.append(torch.cat(tmp_list, 0)[None])


                img_edit = torch.cat(img_edit_list, 0)

                # utils.save_image(img_edit[0, 0, [2,1,0], ...] / 255, 'output.jpg')


                if i == 0:
                    utils.save_image(img_edit.reshape((-1, 3, 256, 704))[:, [2,1,0], ...] / 255, f'./vis_folder/vals_{curr_epoch}.jpg')




                img_input = img_edit - mean
                img_input = img_input / std

                data['img_inputs'][0][0] = img_input

                result = model(return_loss=False, rescale=True, **data)
                # encode mask results
                if isinstance(result[0], tuple):
                    result = [(bbox_results, encode_mask_results(mask_results))
                            for bbox_results, mask_results in result]
            results.extend(result)

            if rank == 0:
                batch_size = len(result)
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        if args.gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), args.tmpdir)

        outputs = results



        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                print(dataset.evaluate(outputs, **eval_kwargs))



def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results



if __name__ == '__main__':
    main()
