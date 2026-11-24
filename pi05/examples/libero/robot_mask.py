"""Calibrated self-mask from a synchronized MuJoCo scene. No sensor depth.

Offline scenes must include all moving object qpos to preserve occlusion.
For live LIBERO, pass the native mjModel/mjData from the same RGB frame.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image


def robot_geom_ids(model, root_names):
    roots=[]
    for name in root_names:
        idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if idx<=0:raise ValueError(f'Robot root must be an existing non-world body: {name}')
        roots.append(idx)
    if not roots:raise ValueError('Explicit robot root body names required')
    bodies=set()
    for b in range(1,model.nbody):
        ancestor=b
        while ancestor:
            if ancestor in roots:
                bodies.add(b);break
            ancestor=int(model.body_parentid[ancestor])
    ids=np.flatnonzero(np.isin(model.geom_bodyid,list(bodies)))
    if not len(ids):raise ValueError('Selected robot subtree contains no geometry')
    return ids


def restore_state(model,data,state):
    # Require all scene coordinates; EEF pose is never treated as joint angles.
    qpos=np.asarray(state['qpos'],dtype=float)
    if qpos.shape!=(model.nq,) or not np.isfinite(qpos).all():
        raise ValueError(f'Require synchronized full scene qpos ({model.nq},), got {qpos.shape}')
    data.qpos[:]=qpos
    if model.nmocap:
        for key,shape in [('mocap_pos',(model.nmocap,3)),('mocap_quat',(model.nmocap,4))]:
            v=np.asarray(state[key],dtype=float)
            if v.shape!=shape or not np.isfinite(v).all():raise ValueError(f'Invalid {key}')
            getattr(data,key)[:]=v
    mujoco.mj_forward(model,data)


def set_external_camera(model,camera_name,calibration,height,width):
    """T_camera_world: world -> OpenCV camera (+x right,+y down,+z forward).
    Currently accepts centered, square-pixel pinhole K only; fail otherwise.
    No silent approximation for cropped/off-center/distorted cameras.
    """
    cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera_name)
    if cid<0:raise ValueError('Missing camera')
    if model.cam_bodyid[cid]!=0 or int(model.cam_mode[cid])!=0:
        raise ValueError('External calibration requires a fixed world-attached camera')
    if calibration['image_size']!=[height,width]:raise ValueError('Calibration size must match raw render size')
    k=np.asarray(calibration['K'],float);t=np.asarray(calibration['T_camera_world'],float)
    if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all():raise ValueError('Invalid calibration')
    if k[0,0]<=0 or not np.allclose(k,[[k[0,0],0,width/2],[0,k[0,0],height/2],[0,0,1]],atol=1e-6):raise ValueError('Only centered square-pixel K supported')
    if not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3]@t[:3,:3].T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(t[:3,:3]),1):raise ValueError('Invalid rigid transform')
    if np.any(np.asarray(calibration.get('distortion',[0]),float)!=0):raise ValueError('Undistort RGB first')
    world_camera=np.linalg.inv(t)
    model.cam_pos[cid]=world_camera[:3,3]
    rotation=world_camera[:3,:3]@np.diag([1.,-1.,-1.])
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,rotation.ravel());model.cam_quat[cid]=quat
    model.cam_fovy[cid]=np.degrees(2*np.arctan(height/(2*k[1,1])))


def render_robot(model,data,camera_name,root_names,height,width,geom_groups):
    """RGB and visible robot pixels from SAME state/camera/geometry settings.
    Renderer handles scene occlusion; object/scene label maps are not exported.
    """
    cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera_name)
    if cid<0:raise ValueError(f'Missing camera {camera_name}')
    if not geom_groups or any(g<0 or g>5 for g in geom_groups):raise ValueError('Explicit visual geom groups 0..5 required')
    ids=robot_geom_ids(model,root_names)
    visible=ids[np.isin(model.geom_group[ids],geom_groups)]
    if not len(visible):raise ValueError('No robot geometry in selected visual groups')
    opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[geom_groups]=1;opt.sitegroup[:]=0
    mujoco.mj_forward(model,data)
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        renderer.update_scene(data,camera=camera_name,scene_option=opt)
        rgb=renderer.render().copy()
    original_samples=int(model.vis.quality.offsamples)
    try:
        model.vis.quality.offsamples=0
        renderer=mujoco.Renderer(model,height=height,width=width)
    finally:
        model.vis.quality.offsamples=original_samples
    with renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data,camera=camera_name,scene_option=opt)
        seg=renderer.render().copy()
    mask=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],ids)
    world_camera=np.eye(4);world_camera[:3,:3]=data.cam_xmat[cid].reshape(3,3)@np.diag([1.,-1.,-1.]);world_camera[:3,3]=data.cam_xpos[cid]
    f=height/(2*np.tan(np.radians(model.cam_fovy[cid])/2))
    metadata={'camera':camera_name,'K':[[f,0,width/2],[0,f,height/2],[0,0,1]],'T_camera_world':np.linalg.inv(world_camera).tolist(),'image_size':[height,width],'robot_root_bodies':root_names,'robot_geom_ids':ids.tolist(),'geom_groups':geom_groups,'sensor_depth_used':False,'scene_occlusion':True,'qpos_sha256':__import__('hashlib').sha256(data.qpos.tobytes()).hexdigest()}
    return rgb,mask,metadata


def align_mask(mask,rotation=0,size=None):
    if rotation not in (0,90,180,270):raise ValueError('Rotation must be explicit 0/90/180/270 degrees counterclockwise')
    out=np.rot90(mask,rotation//90)
    if size is not None:out=np.asarray(Image.fromarray(out).resize((size[1],size[0]),Image.Resampling.NEAREST),dtype=bool)
    return np.ascontiguousarray(out)


def subtract_robot(masks,robot_mask,min_pixels=1):
    masks=np.asarray(masks,dtype=bool);robot_mask=np.asarray(robot_mask,dtype=bool)
    if masks.ndim!=3 or masks.shape[1:]!=robot_mask.shape:raise ValueError('Object masks and robot mask must use identical RGB geometry')
    if min_pixels<1:raise ValueError('min_pixels must be positive')
    clean=masks&~robot_mask[None]
    kept=np.flatnonzero(clean.sum(axis=(1,2))>=min_pixels)
    return clean[kept],kept


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True);p.add_argument('--state',required=True,help='NPZ with full scene qpos, plus mocap arrays if needed')
    p.add_argument('--camera',required=True);p.add_argument('--robot-root',action='append',required=True)
    p.add_argument('--geom-groups',type=int,nargs='+',required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--width',type=int,required=True)
    p.add_argument('--calibration',help='Optional JSON external camera override');p.add_argument('--rotation',type=int,choices=[0,90,180,270],default=0)
    p.add_argument('--output-size',type=int,nargs=2,metavar=('HEIGHT','WIDTH'));p.add_argument('--object-masks',help='Optional NPZ masks in final RGB geometry');p.add_argument('--out',required=True)
    a=p.parse_args();model=mujoco.MjModel.from_xml_path(str(Path(a.scene).resolve()));data=mujoco.MjData(model)
    if a.calibration:set_external_camera(model,a.camera,json.loads(Path(a.calibration).read_text()),a.height,a.width)
    with np.load(a.state) as state:restore_state(model,data,state)
    rgb,mask,meta=render_robot(model,data,a.camera,a.robot_root,a.height,a.width,a.geom_groups)
    mask=align_mask(mask,a.rotation,a.output_size);rgb=np.rot90(rgb,a.rotation//90)
    if a.output_size:rgb=np.asarray(Image.fromarray(rgb).resize((a.output_size[1],a.output_size[0]),Image.Resampling.BILINEAR))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(out/'rendered_rgb.png');Image.fromarray(mask.astype('uint8')*255).save(out/'robot_mask.png')
    np.save(out/'robot_mask.npy',mask);meta.update(rotation_ccw=a.rotation,output_size=list(mask.shape),robot_pixels=int(mask.sum()))
    if a.object_masks:
        with np.load(a.object_masks) as z:
            clean,kept=subtract_robot(z['masks'],mask)
            arrays={k:z[k][kept] if z[k].ndim>0 and len(z[k])==len(z['masks']) else z[k] for k in z.files if k!='masks'}
        np.savez_compressed(out/'clean_objects.npz',masks=clean,source_object_indices=kept,**arrays)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
