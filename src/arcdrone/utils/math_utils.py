from jax import numpy as jp


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians
        
    Returns:
        quaternion: [w, x, y, z]
    """
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


def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion array [w, x, y, z]
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = jp.clip(sinp, -1.0, 1.0)
    pitch = jp.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_error(q_current, q_target):
    """
    Compute quaternion error (angular distance).
    
    Args:
        q_current: Current quaternion [w, x, y, z]
        q_target: Target quaternion [w, x, y, z]
        
    Returns:
        error: Scalar angular error in radians
    """
    # Quaternion difference: q_error = q_target * q_current^-1
    # For unit quaternions, conjugate = inverse
    q_conj = jp.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    
    # Quaternion multiplication: q_target * q_conj
    w = q_target[0] * q_conj[0] - q_target[1] * q_conj[1] - q_target[2] * q_conj[2] - q_target[3] * q_conj[3]
    
    # Angular error from quaternion: angle = 2 * arccos(|w|)
    w_clamped = jp.clip(jp.abs(w), 0.0, 1.0)
    error = 2.0 * jp.arccos(w_clamped)
    
    return error