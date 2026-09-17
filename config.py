"""
config.py — Central configuration for FSOC AI Tracking System
All tunable parameters in one place.
"""
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    width:          int   = 640
    height:         int   = 480
    fov_h_deg:      float = 4.0
    fov_v_deg:      float = 3.0
    fps:            float = 30.0
    pan_speed_max:  float = 16.0   # deg/sec
    tilt_speed_max: float = 16.0
    label:          str   = "Virtual InGaAs/SWIR Sensor Model"


@dataclass
class BeaconConfig:
    intensity:       float = 230.0
    default_size_px: int   = 12     # larger glow = easier detection
    wavelength_nm:   int   = 1550
    fade_rate:       float = 0.02


@dataclass
class SatelliteConfig:
    colour:          int   = 160
    body_w:          int   = 14
    body_h:          int   = 8
    panel_length:    int   = 28
    render_size_px:  int   = 90      # sprite size in pixels


@dataclass
class EarthConfig:
    visible:              bool  = True
    centre_offset_x:      float = -2.5   # world degrees left of centre (Half Earth on left edge)
    centre_offset_y:      float = 0.0    # world degrees
    apparent_radius_deg:  float = 1.5    # apparent radius in FOV degrees


@dataclass
class DisturbanceConfig:
    noise_type:                str   = "none"   # none/salt_pepper/gaussian/poisson/combined
    gaussian_sigma:            float = 5.0
    salt_pepper_ratio:         float = 0.02
    jitter_enabled:            bool  = False
    jitter_max_px:             float = 3.0
    platform_motion_enabled:   bool  = False
    platform_motion_max_px:    float = 4.0
    haze:                      bool  = False
    # HIGH_DYNAMIC_ADAPTIVE support — zero/1.0 = neutral (no effect on existing scenarios)
    extra_jitter_amplitude_px: float = 0.0   # additive multi-tone sinusoidal jitter (px, world-deg via trajectories.py)
    beacon_fade_factor:        float = 1.0   # multiplied into BeaconModel._fade (1.0 = full power)


@dataclass
class TrackingConfig:
    kcf_bbox_size:        int   = 60
    of_max_corners:       int   = 20
    of_win_size:          tuple = (15, 15)


@dataclass
class PIDConfig:
    kp:   float = 8.5    # proportional — aggressive centering response
    ki:   float = 0.25   # integral     — eliminate steady-state offset
    kd:   float = 0.30   # derivative   — active motion damping
    k_ff: float = 0.85   # velocity feedforward gain for zero-lag dynamic trajectory tracking


@dataclass
class TargetsConfig:
    acquisition_time_s:   float = 2.0
    tracking_error_px:    float = 10.0   # PS Mandated threshold: <= 10 px RMSE
    target_loss_pct:      float = 5.0    # PS Mandated threshold: < 5% target loss
    reacquisition_time_s: float = 1.0
    processing_fps:       float = 20.0


@dataclass
class AppConfig:
    camera:      CameraConfig      = field(default_factory=CameraConfig)
    beacon:      BeaconConfig      = field(default_factory=BeaconConfig)
    satellite:   SatelliteConfig   = field(default_factory=SatelliteConfig)
    earth:       EarthConfig       = field(default_factory=EarthConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    tracking:    TrackingConfig    = field(default_factory=TrackingConfig)
    pid:         PIDConfig         = field(default_factory=PIDConfig)
    targets:     TargetsConfig     = field(default_factory=TargetsConfig)


CFG = AppConfig()
