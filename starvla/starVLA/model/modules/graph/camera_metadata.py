"""Optional camera calibration side channel; does not alter image/state/action fields."""

def attach_graph_camera(sample, data_cfg, dataset_name):
    cameras = data_cfg.get('graph_cameras') if data_cfg is not None else None
    if cameras is None:
        return sample
    camera = cameras.get(dataset_name, cameras.get('default'))
    if camera is None:
        raise ValueError(f'Missing graph_cameras calibration for dataset {dataset_name}')
    if camera.get('camera_intrinsics') is None or camera.get('intrinsics_size') is None:
        raise ValueError(f'Populate camera_intrinsics and intrinsics_size for {dataset_name}; sensor depth is never used')
    sample['graph_camera_intrinsics'] = camera['camera_intrinsics']
    sample['graph_intrinsics_size'] = camera['intrinsics_size']
    sample['graph_workspace_image_index'] = int(camera.get('workspace_image_index', 0))
    return sample
