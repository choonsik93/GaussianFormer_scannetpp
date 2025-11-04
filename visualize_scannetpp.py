import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from PIL import Image
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
import open3d as o3d

import warnings
warnings.filterwarnings("ignore")


def draw(voxels, voxel_origin, voxel_size):
    """Visualize the gt or predicted voxel labels.
    
    Args:
        voxel_label (ndarray): The gt or predicted voxel label, with shape (N, 4), N is for number 
            of voxels, 7 is for [x, y, z, label].
        voxel_size (double): The size of each voxel.
        intrinsic (ndarray): The camera intrinsics.
        cam_pose (ndarray): The camera pose.
        d (double): The depth of camera model visualization.
    """
    NYU_COLORS = np.array([
        [ 22, 191, 206, 255], # 00 free
        [214,  38,  40, 255], # 01 ceiling
        [ 43, 160,  43, 255], # 02 floor
        [158, 216, 229, 255], # 03 wall
        [114, 158, 206, 255], # 04 window
        [204, 204,  91, 255], # 05 chair
        [255, 186, 119, 255], # 06 bed
        [147, 102, 188, 255], # 07 sofa
        [ 30, 119, 181, 255], # 08 table
        [188, 188,  33, 255], # 09 tvs
        [255, 127,  12, 255], # 10 furniture
        [196, 175, 214, 255], # 11 objects
        [153, 153, 153, 255], # 12 unknown
    ]).astype(np.uint8)

    def get_grid_coords(dims, resolution):
        """
        :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
        :return coords_grid: is the center coords of voxels in the grid
        """

        g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
        # g_xx = g_xx[::-1]
        g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
        # g_yy = g_yy[::-1]
        g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

        # Obtaining the grid with coords...
        xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
        coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
        coords_grid = coords_grid.astype(np.float32)
        resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

        coords_grid = (coords_grid * resolution) + resolution / 2

        return coords_grid

    grid_coords = get_grid_coords(
        voxels.shape, [voxel_size] * 3
    ) + np.array(voxel_origin[:3], dtype=np.float32).reshape([1, 3])

    voxel_label = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    voxel_label = voxel_label[voxel_label[:, 3] != 0, :]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(voxel_label[:, :3])
    colors = NYU_COLORS[voxel_label[:, 3].astype(np.int32), :3] / 255.0
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size)

    #o3d_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    #o3d_cam_axis.transform(cam_pose)

    o3d.visualization.draw_geometries([o3d_voxel_grid])

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": args.num_samples,
        "vis_scene_index": args.vis_scene_index})

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        try:
            # raw_model.load_state_dict(ckpt['state_dict'], strict=True)
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    if cfg.dataset_type == 'ScannetppDataset':
        miou_metric = MeanIoU(
            list(range(1, 12)),
            12, #17,
            ['ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
               'table', 'tv', 'furniture', 'objects'],
            True, 12, filter_minmax=False)
        miou_metric.reset()
    else:
        miou_metric = MeanIoU(
            list(range(1, 17)),
            17, #17,
            ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'],
            True, 17, filter_minmax=False)
        miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'
    if args.vis_occ or args.vis_gaussian or args.vis_gaussian_topdown:
        save_dir = os.path.join(args.work_dir, f'vis_ep{args.epoch}')
        os.makedirs(save_dir, exist_ok=True)
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            ori_imgs = data.pop('ori_img')
            for i in range(ori_imgs.shape[-1]):
                ori_img = ori_imgs[0, ..., i].cpu().numpy()
                ori_img = ori_img[..., [2, 1, 0]]
                ori_img = Image.fromarray(ori_img.astype(np.uint8))
                ori_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))
            
            # breakpoint()
            result_dict = my_model(imgs=input_imgs, metas=data)
            for idx, pred in enumerate(result_dict['final_occ']):
                pred_occ = pred
                gt_occ = result_dict['sampled_label'][idx]
                # remap 12 -> 0, 0 -> 12
                occ_shape = [40, 40, 16]
                unknown_mask = (pred_occ == 0)
                empty_mask = (pred_occ == 12)
                pred_occ[unknown_mask] = 12
                pred_occ[empty_mask] = 0
                pred_occ = pred_occ.view(1, *occ_shape)
                draw(pred_occ.cpu().numpy()[0], voxel_origin=cfg.pc_range, voxel_size=cfg.grid_size)
                unknown_mask = (gt_occ == 0)
                empty_mask = (gt_occ == 12)
                gt_occ[unknown_mask] = 12
                gt_occ[empty_mask] = 0
                gt_occ = gt_occ.view(1, *occ_shape)
                draw(gt_occ.cpu().numpy()[0], voxel_origin=cfg.pc_range, voxel_size=cfg.grid_size)

                miou_metric._after_step(pred_occ, gt_occ)
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis_gaussian_topdown', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--vis_scene_index', type=int, default=-1)
    parser.add_argument('--vis-scene', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='nusc')
    parser.add_argument('--model-type', type=str, default="base", choices=["base", "prob"])
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
