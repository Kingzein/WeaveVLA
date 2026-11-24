import os
from pathlib import Path

import numpy as np

class RobotProjection:
    """Known robot only, same calibrated render as the validated frontend."""
    def __init__(self, K, E, calibration):
        import pybullet as pb, pkgutil, yaml
        self.pb=pb; self.cid=pb.connect(pb.DIRECT)
        loader=pkgutil.get_loader('eglRenderer')
        plugin=pb.loadPlugin(loader.get_filename(),'_eglRendererPlugin',physicsClientId=self.cid) if loader else pb.loadPlugin('eglRendererPlugin',physicsClientId=self.cid)
        if plugin < 0: raise RuntimeError('Validated EGL robot renderer unavailable')
        env=Path(os.environ['CALVIN_ENV_ROOT'])
        self.cfg=yaml.safe_load((env/'conf/robot/panda.yaml').read_text())
        self.cfg.update(yaml.safe_load((env/'conf/robot/panda_longer_finger.yaml').read_text()))
        position=calibration['robot_base_position']
        orientation=calibration['robot_base_orientation']
        self.uid=pb.loadURDF(str(env/'data'/self.cfg['filename']),position,pb.getQuaternionFromEuler(orientation),useFixedBase=True,physicsClientId=self.cid)
        self.fingers=[i for i in range(pb.getNumJoints(self.uid,physicsClientId=self.cid)) if 'finger' in pb.getJointInfo(self.uid,i,physicsClientId=self.cid)[12].decode().lower()]
        near,far=.01,10.; self.near,self.far=near,far; W=H=200
        pm=np.array([[2*K[0,0]/W,0,(W-2*K[0,2])/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]])
        self.pm=pm.flatten(order='F');self.vm=(np.diag([1,-1,-1,1])@E).flatten(order='F')
    def __call__(self, obs):
        pb=self.pb; cid=self.cid
        for joint,q in zip(self.cfg['arm_joint_ids'],obs[7:14]):pb.resetJointState(self.uid,joint,float(q),physicsClientId=cid)
        for joint in self.cfg['gripper_joint_ids']:pb.resetJointState(self.uid,joint,float(obs[6]/2),physicsClientId=cid)
        im=pb.getCameraImage(200,200,viewMatrix=self.vm,projectionMatrix=self.pm,renderer=pb.ER_BULLET_HARDWARE_OPENGL,flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,physicsClientId=cid)
        seg=np.asarray(im[4]).reshape(200,200);robot=(seg&((1<<24)-1))==self.uid
        depth=np.asarray(im[3]).reshape(200,200)
        return dict(mask=robot&np.isin((seg>>24)-1,self.fingers),robot=robot,depth=self.far*self.near/(self.far-(self.far-self.near)*depth),rgb=np.asarray(im[2]).reshape(200,200,4)[...,:3])
    def close(self):self.pb.disconnect(self.cid)

