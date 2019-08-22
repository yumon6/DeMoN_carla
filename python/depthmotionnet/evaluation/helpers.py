import numpy as np
from minieigen import Matrix3, Vector3, Vector2, Quaternion


def angleaxis_to_angle_axis(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to 
    the angle axis representation with seperate angle and axis.
    aa: minieigen.Vector3
        axis angle with angle as vector magnitude
    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned
    returns the tuple (angle,axis)
    """
    angle = aa.norm()
    if angle < epsilon:
        angle = 0
        axis = Vector3(1,0,0)
    else:
        axis = aa.normalized()
    return angle, axis


def angleaxis_to_quaternion(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to 
    the quaternion representation.
    aa: minieigen.Vector3
        axis angle with angle as vector magnitude
    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned
    returns the unit quaternion
    """
    angle, axis = angleaxis_to_angle_axis(aa,epsilon)
    return Quaternion(angle,axis)



def angleaxis_to_rotation_matrix(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to 
    the rotation matrix representation.
    aa: minieigen.Vector3
        axis angle with angle as vector magnitude
    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned
    returns the 3x3 rotation matrix as numpy.ndarray
    """
    q = angleaxis_to_quaternion(aa,epsilon)
    tmp = q.toRotationMatrix()
    return np.array(tmp)



def motion_vector_to_Rt(motion, epsilon=1e-6):
    """Converts the motion vector to the rotation matrix R and translation t
    motion: np.ndarray
        array with 6 elements. The motions is given as [aa1, aa2, aa3, tx, ty, tz].
        aa1,aa2,aa3 is an angle axis representation. The angle is the norm of the axis.
        [tx, ty, tz] is a 3d translation.
    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned
    returns the 3x3 rotation matrix and the 3d translation vector
    """
    pass
    tmp = motion.squeeze().astype(np.float64)
    t = tmp[3:].copy()
    R = angleaxis_to_rotation_matrix(Vector3(tmp[0:3]),epsilon)
    return R, t


def intrinsics_vector_to_K(intrinsics, width, height):
    """Converts the normalized intrinsics vector to the calibration matrix K
    intrinsics: np.ndarray
        4 element vector with normalized intrinsics [fx, fy, cx, cy]
    width: int
        image width in pixels
    height: int 
        image height in pixels
    returns the calibration matrix K as numpy.ndarray
    """
    tmp = intrinsics.squeeze().astype(np.float64)
    K = np.array([tmp[0]*width, 0, tmp[2]*width, 0, tmp[1]*height, tmp[3]*height, 0, 0, 1], dtype=np.float64).reshape((3,3))
    
    return K
