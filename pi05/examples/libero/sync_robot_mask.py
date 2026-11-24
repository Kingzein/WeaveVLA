"""Visible robot pixels from the SAME scene/settings as robosuite RGB.

Never hide scene objects or override geom groups: both are RGB occluders.
Import before constructing environments so the RGB render stamp is recorded.
"""
import functools
import copy
from types import SimpleNamespace
import hashlib
import mujoco
import numpy as np
from PIL import Image
from robosuite.utils.binding_utils import MjRenderContext
from robot_mask import robot_geom_ids
from weavevla.geometry.contracts import array_digest
from weavevla.geometry.tcp import camera_tcp

VERSION = 'visible_scene_v4_two_fingers_20261020'


def scene_stamp(context):
    """Detect stale RGB, changed poses/cameras, or changed visibility settings."""
    m, d = context.model._model, context.data._data
    h = hashlib.sha256()
    for a in (d.qpos, d.mocap_pos, d.mocap_quat, d.geom_xpos, d.geom_xmat,
              d.cam_xpos, d.cam_xmat, m.cam_fovy, m.geom_group, m.geom_rgba,
              context.vopt.geomgroup, context.vopt.sitegroup):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def model_stamp(context):
    m = context.model._model
    h = hashlib.sha256()
    for a in (m.cam_pos, m.cam_quat, m.cam_fovy, m.geom_pos, m.geom_quat,
              m.geom_group, m.geom_rgba, m.geom_size):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def install_render_guard():
    if getattr(MjRenderContext, '_visible_robot_stamp_v3', False):
        return
    original = MjRenderContext.render
    @functools.wraps(original)
    def render(self, width, height, camera_id=None, segmentation=False):
        self.gl_ctx.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        before = scene_stamp(self)
        result = original(self, width, height, camera_id=camera_id, segmentation=segmentation)
        if scene_stamp(self) != before:
            raise RuntimeError('Scene mutated while rendering RGB/mask')
        if not segmentation:
            stamps = getattr(self, '_visible_robot_rgb_stamps', {})
            stamps[int(self.cam.fixedcamid)] = dict(width=int(width), height=int(height),
                stamp=before, model_stamp=model_stamp(self), data=copy.copy(self.data._data),
                options=copy.copy(self.vopt))
            self._visible_robot_rgb_stamps = stamps
        return result
    MjRenderContext.render = render
    original_read = MjRenderContext.read_pixels
    @functools.wraps(original_read)
    def read_pixels(self, width, height, depth=False, segmentation=False):
        self.gl_ctx.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        result = original_read(self, width, height, depth=depth, segmentation=segmentation)
        if not segmentation:
            rgb = result[0] if depth else result
            snapshot = getattr(self, '_visible_robot_rgb_stamps', {}).get(int(self.cam.fixedcamid))
            if snapshot is None: raise RuntimeError('RGB read without a matching render snapshot')
            primary = np.asarray(Image.fromarray(rgb[::-1, ::-1]).resize((224,224),Image.Resampling.BILINEAR))
            snapshot['rgb_sha256'] = array_digest(primary)
        return result
    MjRenderContext.read_pixels = read_pixels
    MjRenderContext._visible_robot_stamp_v3 = True


install_render_guard()


class LiveRobotMask:
    version = VERSION

    def __init__(self, env):
        self.env = env
        self.sim = env.env.sim
        self.model = self.sim.model._model
        self.context = self.sim._render_context_offscreen
        robot = env.env.robots[0]
        self.ids = robot_geom_ids(self.model, [robot.robot_model.root_body,
                                              robot.robot_model.mount.root_body])
        # Exact robot-specific names; fail rather than silently use the palm.
        prefix = robot.gripper.naming_prefix
        self.finger_bodies = [prefix+'leftfinger', prefix+'rightfinger']
        for name in self.finger_bodies:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                raise RuntimeError('Unsupported gripper: missing finger body '+name)
        self.gripper_ids = robot_geom_ids(self.model, self.finger_bodies)
        if not len(self.gripper_ids):
            raise RuntimeError('No finger geometry')
        if not np.isin(self.gripper_ids, self.ids).all():
            raise RuntimeError('Gripper geometry is not in the current robot subtree')
        self.camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'agentview')
        if self.camera < 0:
            raise ValueError('agentview camera absent')
        # Preserve the existing intrinsics convention; unrelated calibration is unchanged.
        f = 256 / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera]) / 2))
        self.K = np.array([[f, 0, 127], [0, f, 127], [0, 0, 1]])
        self.last_metadata = None
        # ID colors must not be averaged by MSAA: blended colors can decode to
        # unrelated geometry IDs (including robot IDs on foreground objects).
        # Keep RGB's context/settings intact and use a dedicated zero-MSAA
        # framebuffer in the SAME EGL context for segmentation only.
        self.context.gl_ctx.make_current()
        samples = int(self.model.vis.quality.offsamples)
        try:
            self.model.vis.quality.offsamples = 0
            self.mask_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        finally:
            self.model.vis.quality.offsamples = samples
        if self.mask_context.offSamples != 0:
            raise RuntimeError('Segmentation framebuffer unexpectedly uses multisampling')

    def render(self, rgb):
        if self.env is None: raise RuntimeError('Robot mask renderer is closed')
        if (self.env.env.sim is not self.sim or
                self.env.env.sim.model._model is not self.model or
                self.sim._render_context_offscreen is not self.context):
            raise RuntimeError('Simulator reset/replaced: rebuild LiveRobotMask before use')
        snapshot = getattr(self.context, '_visible_robot_rgb_stamps', {}).get(self.camera)
        if snapshot is None or (snapshot['width'], snapshot['height']) != (256, 256):
            raise RuntimeError('Robot mask refused: no synchronized 256x256 agentview RGB; refresh observations')
        if snapshot.get('rgb_sha256') != array_digest(np.asarray(rgb)):
            raise RuntimeError('Robot mask refused: supplied RGB is not the captured RGB observation')
        if snapshot['model_stamp'] != model_stamp(self.context):
            raise RuntimeError('Model/visibility changed since RGB: refresh observations before mask')
        # robosuite can advance physics after updating image sensors. Use the actual
        # RGB render's mjData snapshot, not the slightly newer end-of-step state.
        # No scene geometry is hidden; preserve exactly the RGB visibility options.
        original_data, original_options, original_con = self.context.data, self.context.vopt, self.context.con
        try:
            self.context.data = SimpleNamespace(_data=snapshot['data'])
            self.context.vopt = snapshot['options']
            self.context.con = self.mask_context
            seg = self.sim.render(256, 256, camera_name='agentview', segmentation=True)
            if scene_stamp(self.context) != snapshot['stamp']:
                raise RuntimeError('Scene snapshot changed between RGB and segmentation')
        finally:
            self.context.data, self.context.vopt, self.context.con = original_data, original_options, original_con
            self.context.gl_ctx.make_current()
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, original_con)
        before = snapshot['stamp']
        if seg.shape != (256, 256, 2):
            raise RuntimeError('Unexpected robosuite segmentation format')
        valid = seg[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = (valid & np.isin(seg[:, :, 1], self.ids))[::-1, ::-1]
        grip = (valid & np.isin(seg[:, :, 1], self.gripper_ids))[::-1, ::-1]
        mask = np.asarray(Image.fromarray(mask).resize((224, 224), Image.Resampling.NEAREST), dtype=bool)
        grip = np.asarray(Image.fromarray(grip).resize((224, 224), Image.Resampling.NEAREST), dtype=bool)
        if np.any(grip & ~mask):
            raise RuntimeError('Gripper mask exceeds robot mask')
        self.last_metadata = dict(version=VERSION, gripper_scope="two_finger_subtrees_only", finger_bodies=self.finger_bodies, scene_sha256=before,
                                  scene_occlusion=True, rgb_state_verified=True,
                                  robot_pixels=int(mask.sum()), gripper_pixels=int(grip.sum()),
                                  segmentation_msaa_samples=0,rgb_sha256=snapshot['rgb_sha256'],robot_mask_sha256=array_digest(mask),gripper_mask_sha256=array_digest(grip))
        self.last_metadata['tcp'] = camera_tcp(self.model, self.sim.data._data)
        return mask, grip, self.K

    def close(self):
        # Release borrowed renderer references before the next environment is
        # constructed. A delayed MuJoCo GL destructor can otherwise run while
        # the next environment's context is current and corrupt its buffers.
        if getattr(self, 'mask_context', None) is not None:
            self.context.gl_ctx.make_current()
            self.mask_context.free()
            self.mask_context = None
        self.env = self.sim = self.model = self.context = None
