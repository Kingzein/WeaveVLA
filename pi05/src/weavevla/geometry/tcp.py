"""Robot-state TCP in the SAME image-aligned metric chart as DA3 points.

Raw robosuite/OpenGL image is bottom-up, then this pipeline rotates both axes.
Thus image-aligned coordinates are (-x_gl,-y_gl,-z_gl). This reflected chart
must not be confused with the conventional unmirrored OpenCV camera frame.
"""
import numpy as np
TCP_VERSION='libero_grip_site_image_aligned_v1'

def camera_tcp(model,data,site_name='gripper0_grip_site',camera_name='agentview'):
 import mujoco
 sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site_name)
 cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera_name)
 if sid<0 or cid<0:raise ValueError('Explicit TCP site and camera required; unsupported robot mapping')
 point=np.asarray(data.site_xpos[sid],np.float64)
 rotation=np.asarray(data.cam_xmat[cid]).reshape(3,3)
 T=np.eye(4);T[:3,:3]=-rotation.T;T[:3,3]=rotation.T@data.cam_xpos[cid]
 p=(T@np.r_[point,1])[:3]
 if not np.isfinite(p).all() or p[2]<=0:raise ValueError('Invalid camera-frame TCP')
 return dict(version=TCP_VERSION,position_camera=p.tolist(),position_world=point.tolist(),T_image_camera_world=T.tolist(),site=site_name,camera=camera_name,source='robot_forward_kinematics',depth_scaled=False,image_transform='raw_opengl_rotate180',coordinate_handedness='reflected_image_chart')

def position(record):
 if not isinstance(record,dict) or record.get('version')!=TCP_VERSION or record.get('depth_scaled') is not False:
  raise ValueError('Matching unscaled TCP record required')
 p=np.asarray(record['position_camera'],np.float32)
 if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Invalid TCP position')
 return p
