# ================== data ========================
occ_path = "data/surroundocc/samples"

input_shape = (640, 480)
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

dataset_type = 'ScannetppDataset'
data_root = '/data'
class_names = ('ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
               'table', 'tv', 'furniture', 'objects')

metainfo = dict(classes=class_names,
                occ_classes=class_names,
                box_type_3d='euler-depth')
backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(640, 480), keep_ratio=False),
#     dict(type='Normalize', **img_norm_cfg),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(640, 480), keep_ratio=False),
#     dict(type='Normalize', **img_norm_cfg),
# ]

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancyScannetpp", occ_path=occ_path, semantic=True, use_ego=False, voxel_dims=(40, 40, 16)),
    dict(type="ScannetppResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadOccupancyScannetpp", occ_path=occ_path, semantic=True, use_ego=False, voxel_dims=(40, 40, 16)),
    dict(type="ScannetppResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 1168,
    "W": 1752,
    "rand_flip": False,
}

# train_dataloader = dict(batch_size=1,
#                         num_workers=1,
#                         persistent_workers=True,
#                         sampler=dict(type='DefaultSampler', shuffle=True),
#                         dataset=dict(type=dataset_type,
#                                      data_root=data_root,
#                                      ann_file='scannetpp_infos_train.pkl',
#                                      pipeline=train_pipeline,
#                                      test_mode=False,
#                                      filter_empty_gt=True,
#                                      box_type_3d='Euler-Depth',
#                                      metainfo=metainfo))

# val_dataloader = dict(batch_size=1,
#                       num_workers=1,
#                       persistent_workers=True,
#                       drop_last=False,
#                       sampler=dict(type='DefaultSampler', shuffle=False),
#                       dataset=dict(type=dataset_type,
#                                    data_root=data_root,
#                                    ann_file='scannetpp_infos_val.pkl',
#                                    pipeline=test_pipeline,
#                                    test_mode=True,
#                                    filter_empty_gt=True,
#                                    box_type_3d='Euler-Depth',
#                                    metainfo=metainfo))
# test_dataloader = val_dataloader

train_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + '/scannetpp_infos_2x_train.pkl',
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
    metainfo=metainfo,
)

val_dataset_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + '/scannetpp_infos_2x_val.pkl',
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
    metainfo=metainfo,
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2
)