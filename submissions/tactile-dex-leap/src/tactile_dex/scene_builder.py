"""Build the TACTILE-DEX MuJoCo scene around the LEAP Hand.

The menagerie ``right_hand.xml`` only allows a single ``<sensor>`` element
(already populated with 16 ``jointpos`` sensors) and ships no fingertip
``<site>``s. Rather than editing the upstream asset, we load it through
``mujoco.MjSpec`` and *programmatically* attach touch sensors and a manipulated
cube, so the upstream LEAP files stay verbatim and easy to reproduce.

Layout
------
The menagerie ships the palm at ``quat="0 1 0 0"`` (180 deg about x), which
makes the fingers curl *upward* when flexed. We reset the palm to identity
orientation so the open hand faces up and the fingers curl down onto the cube.
The cube rests on a small table beneath the fingers; closing the hand wraps the
fingertips around it (gravity pins the cube, which makes shaping far easier).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# Fingertip distal bodies (from the menagerie right_hand.xml).
FINGERTIP_BODIES = ("if_ds", "mf_ds", "rf_ds", "th_ds")
FINGER_NAMES = ("index", "middle", "ring", "thumb")

# Local-frame offset from each fingertip *body origin* to the centre of the
# tip geom (the menagerie ``tip``/``thumb_tip`` mesh geoms carry an intrinsic
# origin offset, so the geom centre is NOT at the body origin). These were
# measured once from the compiled model (geom_xpos - body_xpos, rotated into
# the body frame) and let the touch site sit on the actual contact pad.
TIP_SITE_OFFSET = {
    "if_ds": (0.0089, -0.0247, -0.0102),
    "mf_ds": (0.0089, -0.0247, -0.0102),
    "rf_ds": (0.0089, -0.0247, -0.0102),
    "th_ds": (0.0456, -0.0013, 0.0),
}


@dataclass
class SceneConfig:
    """Tunable scene parameters (kept in one place so the env and demo agree)."""

    leap_xml: Path = Path(__file__).resolve().parents[2] / "assets" / "leap_hand" / "right_hand.xml"
    timestep: float = 0.002
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Palm mount. Identity orientation (instead of the menagerie default
    # ``0 1 0 0``) makes the fingers curl down onto the cube.
    palm_pos: tuple[float, float, float] = (0.0, 0.0, 0.1)
    palm_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Manipulated object. "cube" is the training object; the demo can switch to
    # sphere / cylinder / bottle to show the hand generalizing across shapes.
    object_type: str = "cube"  # cube | sphere | cylinder | bottle
    cube_size: float = 0.022  # half-side (cube) / radius (sphere) / etc.
    cube_mass: float = 0.060
    cube_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    # Fingertip contact tuning (grippy rubber-on-plastic).
    tip_friction: tuple[float, float, float] = (1.5, 0.005, 0.0001)
    tip_solref: tuple[float, float] = (0.02, 1.0)
    tip_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001)
    tip_condim: int = 6

    # Cube spawn pose (world frame). The open LEAP fingers span x~0.04→0.12 at
    # z~0.08, forming a "tray". We drop the cube just above that tray
    # (centre z~0.10) so it settles onto the fingers, then closing the hand
    # curls the fingertips down around it.
    cube_spawn_pos: tuple[float, float, float] = (0.07, 0.0, 0.10)
    table_z: float = 0.0  # safety floor (cube is caught by the fingers, not the table)

    render_width: int = 1280
    render_height: int = 720

    # Optional "grasp assist" weld between cube and palm. The env toggles it per
    # episode for the curriculum; the CurriculumCallback in train.py anneals
    # weld_steps (rigid weld length) from long -> 0 over the first training window.
    use_grasp_weld: bool = True


def _add_fingertip_touch(spec: mujoco.MjSpec, body_name: str, finger_label: str) -> None:
    """Attach a touch site + sensor to one fingertip body."""
    body = spec.body(body_name)
    if body is None:
        raise ValueError(f"LEAP body '{body_name}' missing — wrong right_hand.xml?")

    site_name = f"{finger_label}_touch_site"
    body.add_site(
        name=site_name,
        pos=list(TIP_SITE_OFFSET[body_name]),
        # Generous radius so any contact on the pad falls inside the sensor
        # region (touch = sum of normal forces for contacts whose point is
        # within the site's sphere).
        size=[0.012, 0.012, 0.012],
        rgba=[0.2, 1.0, 0.2, 0.6],
        group=2,
    )
    spec.add_sensor(
        name=f"{finger_label}_touch",
        type=mujoco.mjtSensor.mjSENS_TOUCH,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=site_name,
    )


def _tune_fingertip_friction(model: mujoco.MjModel) -> None:
    """Override friction/solref/solimp/condim on the four tip geoms.

    In the compiled model ``geom_friction`` is a 3-vector (slide1, slide2,
    torsion) and ``geom_solimp`` is a 5-vector; we keep them in sync with the
    scene config so the env and the rendered scene agree.
    """
    cfg_tip_names = ("if_tip", "mf_tip", "rf_tip", "th_tip")
    for name in cfg_tip_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        model.geom_friction[gid] = (1.5, 0.005, 0.0001)
        model.geom_solref[gid] = (0.02, 1.0)
        model.geom_solimp[gid] = (0.9, 0.95, 0.001, 0.5, 2.0)
        model.geom_condim[gid] = 6


def build_scene(config: SceneConfig | None = None) -> tuple[mujoco.MjModel, dict]:
    """Compile the full TACTILE-DEX model.

    Returns the compiled ``MjModel`` plus a dict of useful name/id maps.
    """
    config = config or SceneConfig()
    if not config.leap_xml.exists():
        raise FileNotFoundError(f"LEAP right_hand.xml not found at {config.leap_xml}")

    spec = mujoco.MjSpec.from_file(str(config.leap_xml))
    spec.option.timestep = config.timestep
    spec.option.gravity = list(config.gravity)
    spec.visual.global_.offwidth = config.render_width
    spec.visual.global_.offheight = config.render_height

    # Reposition the palm so the open hand faces up (fingers curl down onto
    # the cube). The menagerie default "0 1 0 0" curls fingers upward.
    palm = spec.body("palm")
    if palm is None:
        raise ValueError("LEAP 'palm' body missing — wrong right_hand.xml?")
    palm.pos = list(config.palm_pos)
    palm.quat = list(config.palm_quat)

    # 1) Touch sensors at every fingertip.
    for body_name, label in zip(FINGERTIP_BODIES, FINGER_NAMES):
        _add_fingertip_touch(spec, body_name, label)

    # 2) Floor as a visual ground plane + drop safety net. The cube is cradled
    # by the fingers, so we do not add a workbench that would collide with the
    # hand while it closes.
    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, config.table_z],
        rgba=[0.08, 0.09, 0.11, 1.0],
    )

    # 3) The manipulated object (free-floating body with a free joint).
    obj_type = config.object_type
    s = config.cube_size
    if obj_type == "sphere":
        geom_kw = dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[s])
    elif obj_type == "cylinder":
        geom_kw = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[s * 0.8, s * 1.4])
    elif obj_type == "bottle":
        # a capsule reads as a bottle/pen — graspable along its long axis.
        geom_kw = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[s * 0.7, s * 1.8])
    else:  # cube (default)
        geom_kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[s, s, s])
    cube = world.add_body(
        name="cube",  # name kept as "cube" so the env indexing is unchanged
        pos=list(config.cube_spawn_pos),
    )
    cube.add_freejoint(name="cube_freejoint")
    cube.add_geom(
        name="cube_geom",
        mass=config.cube_mass,
        friction=list(config.cube_friction),
        condim=3,
        solref=(0.02, 1.0),
        solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        rgba=[1.0, 0.55, 0.15, 1.0],
        **geom_kw,
    )
    # Coloured target ghost (visual-only) so the demo can show the goal pose.
    target = world.add_body(
        name="target_ghost",
        pos=list(config.cube_spawn_pos),
    )
    target.add_geom(
        name="target_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[config.cube_size, config.cube_size, config.cube_size],
        contype=0,
        conaffinity=0,
        rgba=[0.2, 0.9, 0.4, 0.25],
    )

    # 4) Optional grasp weld (cube <-> palm) for the scripted baseline.
    # WELD eq_data layout (11 floats): anchor(3) + relpose quat(4) + relpose pos(3)
    # + torquescale(1). We anchor at the cube origin and let MuJoCo infer relpose
    # at compile; passing zeros leaves the defaults MuJoCo fills in from body poses.
    if config.use_grasp_weld:
        spec.add_equality(
            name="grasp_weld",
            type=mujoco.mjtEq.mjEQ_WELD,
            name1="cube",
            name2="palm",
            objtype=mujoco.mjtObj.mjOBJ_BODY,
        )

    # 5) Lighting + cameras.
    world.add_light(pos=[0.2, -0.3, 0.6], dir=[-0.2, 0.3, -1.0], diffuse=[0.9, 0.9, 0.9])
    world.add_light(pos=[-0.3, 0.2, 0.5], dir=[0.3, -0.2, -1.0], diffuse=[0.5, 0.55, 0.65])
    world.add_camera(
        name="hero",
        pos=[0.18, -0.22, 0.16],
        xyaxes=[-0.6, 0.8, 0.0, -0.25, -0.18, 0.95],
    )
    world.add_camera(
        name="front",
        pos=[0.0, -0.30, 0.12],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.3, 0.95],
    )

    model = spec.compile()
    _tune_fingertip_friction(model)

    # Disable the weld by default — the env re-enables it only for the scripted
    # baseline. eq_active0 is the design-time default.
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    if eq_id >= 0:
        model.eq_active0[eq_id] = 0

    # Index maps the env/demo will reuse.
    info = {
        "fingertip_bodies": FINGERTIP_BODIES,
        "finger_names": FINGER_NAMES,
        "touch_sensor_names": [f"{n}_touch" for n in FINGER_NAMES],
        "cube_body": "cube",
        "cube_geom": "cube_geom",
        "palm_body": "palm",
        "target_body": "target_ghost",
        "actuator_names": [
            "if_mcp_act", "if_rot_act", "if_pip_act", "if_dip_act",
            "mf_mcp_act", "mf_rot_act", "mf_pip_act", "mf_dip_act",
            "rf_mcp_act", "rf_rot_act", "rf_pip_act", "rf_dip_act",
            "th_cmc_act", "th_axl_act", "th_mcp_act", "th_ipl_act",
        ],
        "joint_names": [
            "if_mcp", "if_rot", "if_pip", "if_dip",
            "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
            "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
            "th_cmc", "th_axl", "th_mcp", "th_ipl",
        ],
    }
    return model, info


def set_grasp_weld(model: mujoco.MjModel, data: mujoco.MjData, active: bool) -> None:
    """Toggle the soft grasp weld at runtime."""
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    if eq_id >= 0:
        data.eq_active[eq_id] = 1 if active else 0


def touch_values(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the 4 fingertip touch readings (Newtons), ordered like FINGER_NAMES."""
    out = np.zeros(len(FINGER_NAMES), dtype=np.float64)
    for i, name in enumerate(FINGER_NAMES):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{name}_touch")
        if sid < 0:
            continue
        adr = model.sensor_adr[sid]
        out[i] = data.sensordata[adr]
    return out
