"""
CIS PA1 - 3D Cartesian Math Package
Computer Integrated Surgery Programming Assignment 1

This module provides 3D mathematical operations for points, rotations, 
and frame transformations required for electromagnetic tracking system calibration.

Authors: Rohit Satisha and Sahana Raja

LIBRARIES USED:
- NumPy (https://numpy.org/): For basic linear algebra operations (SVD, matrix operations)
  Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
"""

import numpy as np
from typing import Tuple, List, Optional
import math


class Point3D:
    """3D Point class with basic vector operations."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Point3D':
        return self.__mul__(scalar)
    
    def dot(self, other: 'Point3D') -> float:
        """Dot product of two 3D points."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Point3D') -> 'Point3D':
        """Cross product of two 3D points."""
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def norm(self) -> float:
        """Euclidean norm (magnitude) of the vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Point3D':
        """Return normalized version of the vector."""
        n = self.norm()
        if n == 0:
            return Point3D(0, 0, 0)
        return Point3D(self.x/n, self.y/n, self.z/n)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point3D':
        """Create Point3D from numpy array."""
        return cls(arr[0], arr[1], arr[2])
    
    def __repr__(self) -> str:
        return f"Point3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


class Rotation3D:
    """3D Rotation class with various rotation representations."""
    
    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self.matrix = np.eye(3)
        else:
            self.matrix = matrix.copy()
    
    @classmethod
    def from_axis_angle(cls, axis: Point3D, angle: float) -> 'Rotation3D':
        """Create rotation from axis-angle representation using Rodrigues' formula."""
        axis = axis.normalize()
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        one_minus_cos = 1 - cos_angle
        
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        K = np.array([
            [0, -axis.z, axis.y],
            [axis.z, 0, -axis.x],
            [-axis.y, axis.x, 0]
        ])
        
        R = np.eye(3) + sin_angle * K + one_minus_cos * np.dot(K, K)
        return cls(R)
    
    @classmethod
    def from_euler_angles(cls, alpha: float, beta: float, gamma: float, 
                         order: str = 'xyz') -> 'Rotation3D':
        """Create rotation from Euler angles."""
        Rx = cls._rotation_x(alpha)
        Ry = cls._rotation_y(beta)
        Rz = cls._rotation_z(gamma)
        
        if order == 'xyz':
            R = np.dot(Rz, np.dot(Ry, Rx))
        elif order == 'zyz':
            R = np.dot(Rz, np.dot(Ry, Rz))
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
        
        return cls(R)
    
    @classmethod
    def from_quaternion(cls, q0: float, q1: float, q2: float, q3: float) -> 'Rotation3D':
        """Create rotation from quaternion (w, x, y, z)."""
        # Normalize quaternion
        norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm
        
        # Convert to rotation matrix
        R = np.array([
            [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
        ])
        
        return cls(R)
    
    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        """Rotation matrix about x-axis."""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        """Rotation matrix about y-axis."""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Rotation matrix about z-axis."""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def apply(self, point: Point3D) -> Point3D:
        """Apply rotation to a 3D point."""
        p_array = point.to_array()
        rotated = np.dot(self.matrix, p_array)
        return Point3D.from_array(rotated)
    
    def inverse(self) -> 'Rotation3D':
        """Get inverse rotation (transpose of rotation matrix)."""
        return Rotation3D(self.matrix.T)
    
    def compose(self, other: 'Rotation3D') -> 'Rotation3D':
        """Compose this rotation with another rotation."""
        return Rotation3D(np.dot(self.matrix, other.matrix))
    
    def determinant(self) -> float:
        """Get determinant of rotation matrix (should be +1)."""
        return np.linalg.det(self.matrix)
    
    def to_axis_angle(self) -> Tuple[Point3D, float]:
        """Convert rotation matrix to axis-angle representation."""
        # Extract axis and angle from rotation matrix
        trace = np.trace(self.matrix)
        angle = math.acos(max(-1, min(1, (trace - 1) / 2)))
        
        if abs(angle) < 1e-6:
            # Identity rotation
            return Point3D(1, 0, 0), 0.0
        
        # Extract axis
        axis = Point3D(
            self.matrix[2, 1] - self.matrix[1, 2],
            self.matrix[0, 2] - self.matrix[2, 0],
            self.matrix[1, 0] - self.matrix[0, 1]
        ).normalize()
        
        return axis, angle
    
    def __repr__(self) -> str:
        return f"Rotation3D(\n{self.matrix}\n)"


class Frame3D:
    """3D Frame transformation class combining rotation and translation."""
    
    def __init__(self, rotation: Rotation3D = None, translation: Point3D = None):
        self.rotation = rotation if rotation is not None else Rotation3D()
        self.translation = translation if translation is not None else Point3D()
    
    @classmethod
    def identity(cls) -> 'Frame3D':
        """Create identity frame transformation."""
        return cls()
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Frame3D':
        """Create frame from 4x4 homogeneous transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        
        rotation = Rotation3D(matrix[:3, :3])
        translation = Point3D(matrix[0, 3], matrix[1, 3], matrix[2, 3])
        return cls(rotation, translation)
    
    def apply(self, point: Point3D) -> Point3D:
        """Apply frame transformation to a 3D point."""
        rotated = self.rotation.apply(point)
        return rotated + self.translation
    
    def inverse(self) -> 'Frame3D':
        """Get inverse frame transformation."""
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply(Point3D() - self.translation)
        return Frame3D(inv_rotation, inv_translation)
    
    def compose(self, other: 'Frame3D') -> 'Frame3D':
        """Compose this frame with another frame."""
        new_rotation = self.rotation.compose(other.rotation)
        new_translation = self.rotation.apply(other.translation) + self.translation
        return Frame3D(new_rotation, new_translation)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.matrix
        matrix[:3, 3] = [self.translation.x, self.translation.y, self.translation.z]
        return matrix
    
    def __repr__(self) -> str:
        return f"Frame3D(R={self.rotation}, t={self.translation})"


def compute_centroid(points: List[Point3D]) -> Point3D:
    """Compute centroid of a list of 3D points."""
    if not points:
        return Point3D()
    
    sum_x = sum(p.x for p in points)
    sum_y = sum(p.y for p in points)
    sum_z = sum(p.z for p in points)
    n = len(points)
    
    return Point3D(sum_x/n, sum_y/n, sum_z/n)


def compute_covariance_matrix(points_a: List[Point3D], points_b: List[Point3D]) -> np.ndarray:
    """Compute covariance matrix for point set registration."""
    if len(points_a) != len(points_b):
        raise ValueError("Point sets must have same length")
    
    n = len(points_a)
    H = np.zeros((3, 3))
    
    for i in range(n):
        a = points_a[i].to_array()
        b = points_b[i].to_array()
        H += np.outer(a, b)
    
    return H


def skew_symmetric_matrix(point: Point3D) -> np.ndarray:
    """Create skew-symmetric matrix from 3D point."""
    return np.array([
        [0, -point.z, point.y],
        [point.z, 0, -point.x],
        [-point.y, point.x, 0]
    ])


# Test the implementation
if __name__ == "__main__":
    # Test basic operations
    p1 = Point3D(1, 2, 3)
    p2 = Point3D(4, 5, 6)
    
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Dot product: {p1.dot(p2)}")
    print(f"Cross product: {p1.cross(p2)}")
    print(f"Norm of p1: {p1.norm()}")
    
    # Test rotation
    axis = Point3D(0, 0, 1)  # z-axis
    angle = math.pi / 2  # 90 degrees
    R = Rotation3D.from_axis_angle(axis, angle)
    
    test_point = Point3D(1, 0, 0)
    rotated = R.apply(test_point)
    print(f"Rotated (1,0,0) by 90° around z-axis: {rotated}")
    
    # Test frame transformation
    frame = Frame3D(R, Point3D(1, 1, 1))
    transformed = frame.apply(Point3D(0, 0, 0))
    print(f"Frame transformation of origin: {transformed}")
