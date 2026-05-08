"""Math utilities — local copy for New_attempt_2 (self-contained)."""

from jax import numpy as jp


def euler_to_quaternion(roll, pitch, yaw):
    cy = jp.cos(yaw * 0.5)
    sy = jp.sin(yaw * 0.5)
    cp = jp.cos(pitch * 0.5)
    sp = jp.sin(pitch * 0.5)
    cr = jp.cos(roll * 0.5)
    sr = jp.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return jp.array([w, x, y, z])
