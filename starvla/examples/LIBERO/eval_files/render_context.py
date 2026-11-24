"""Bind each robosuite renderer's own EGL context before drawing/readback.

Scoped to the Plus evaluator process; never modifies the installed simulator.
"""
import functools
import mujoco
from robosuite.utils.binding_utils import MjRenderContext


def install():
    if getattr(MjRenderContext, '_weavevla_context_bound', False):
        return
    for name in ('render', 'read_pixels'):
        original = getattr(MjRenderContext, name)
        def bind(original):
            @functools.wraps(original)
            def call(self, *args, **kwargs):
                self.gl_ctx.make_current()
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
                return original(self, *args, **kwargs)
            return call
        setattr(MjRenderContext, name, bind(original))
    MjRenderContext._weavevla_context_bound = True
