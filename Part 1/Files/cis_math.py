"""
CIS PA1 - 3D Cartesian Math Package
Authors: Rohit Satish and Sahana Raja


This module provides 3D mathematical operations for:
  - Points 
  - Rotations 
  - Frame transformations 
These are all required for electromagnetic tracking system calibration.


LIBRARIES USED:
- NumPy (https://numpy.org/): For basic linear algebra operations (SVD, matrix operations)
  Citation: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). https://doi.org/10.1038/s41586-020-2649-2
"""

import numpy as np
from typing import Tuple, List, Optional
import math

#Defines the attributes of a 3D point alogn with result of basic 3D operations
class Point3D:

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """3D point initialization that essentially says the object or self has an x,y, and z attributable to itself

        Args:
            x (float): X-coordinate for the point.
            y (float): Y-coordinate for the point.
            z (float): Z-coordinate for the point.

        Returns:
            None: This initializer configures the point in place.
        """
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: 'Point3D') -> 'Point3D':
        """Function that is known to return another Point3D that adds two Point3D's together

        Args:
            other (Point3D): Point to add to this point.

        Returns:
            Point3D: Resulting point after vector addition.
        """
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Point3D') -> 'Point3D':
        """Function that is known to return another Point3D that subtracts two Point3D's

        Args:
            other (Point3D): Point to subtract from this point.

        Returns:
            Point3D: Resulting point after vector subtraction.
        """
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Point3D':
        """Function that multiplies a Point3D by a specified scalar

        Args:
            scalar (float): Scalar value used to scale each component.

        Returns:
            Point3D: Scaled point.
        """
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)
      
    def __rmul__(self, scalar: float) -> 'Point3D':
        """Handles the case of the scalar being on the left side of the vector

        Args:
            scalar (float): Scalar value used to scale each component.

        Returns:
            Point3D: Scaled point.
        """
        return self.__mul__(scalar)

    def dot(self, other: 'Point3D') -> float:
        """Dot Product of two Point3D's

        Args:
            other (Point3D): Point used for the dot product.

        Returns:
            float: Scalar dot product of the two points.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Point3D') -> 'Point3D':
        """Cross Product of two Point3D's

        Args:
            other (Point3D): Point used for the cross product.

        Returns:
            Point3D: Cross product vector of the two points.
        """
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def norm(self) -> float:
        """Euclidean norm (magnitude) of the vector

        Args:
            None: This method uses the point's own coordinates.

        Returns:
            float: Euclidean norm of the point vector.
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Point3D':
        """Returns normalized version of the vector which means the final vector is a unit vector

        Args:
            None: This method uses the point's own coordinates.

        Returns:
            Point3D: Unit vector pointing in the same direction.
        """
        n = self.norm()
        if n == 0:
            return Point3D(0, 0, 0)
        return Point3D(self.x/n, self.y/n, self.z/n)
    
    def to_array(self) -> np.ndarray:
        """Converts a Point3D to numpy array

        Args:
            None: This method uses the point's own coordinates.

        Returns:
            np.ndarray: Array representation of the point.
        """
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point3D':
        """Create Point3D from numpy array.

        Args:
            arr (np.ndarray): Array containing three coordinates.

        Returns:
            Point3D: Point constructed from the array values.
        """
        return cls(arr[0], arr[1], arr[2])
    
    def __repr__(self) -> str:
        """toString method

        Args:
            None: This method uses the point's own coordinates.

        Returns:
            str: String representation of the point.
        """
        return f"Point3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

#3D Rotation class with various rotation representations.
class Rotation3D:
    def __init__(self, matrix: Optional[np.ndarray] = None):
        """Initializes the rotation matrix for the 3D rotation object.

        Args:
            matrix (Optional[np.ndarray]): Rotation matrix used to initialize the object.

        Returns:
            None: This initializer configures the rotation in place.
        """
        if matrix is None:
            self.matrix = np.eye(3) # Initializes as identity matrix if matrix not given
        else:
            self.matrix = matrix.copy()
    
    @classmethod
    def from_axis_angle(cls, axis: Point3D, angle: float) -> 'Rotation3D':
        """Create rotation from axis-angle representation using Rodrigues' formula.

        Args:
            axis (Point3D): Rotation axis.
            angle (float): Rotation angle in radians.

        Returns:
            Rotation3D: Rotation constructed from the axis-angle representation.
        """
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
        """Create rotation from Euler angles

        Args:
            alpha (float): Rotation angle about the x-axis.
            beta (float): Rotation angle about the y-axis.
            gamma (float): Rotation angle about the z-axis.
            order (str): Order in which rotations are applied.

        Returns:
            Rotation3D: Rotation constructed from Euler angles.
        """
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
        """Create rotation from quaternion (w, x, y, z)

        Args:
            q0 (float): Scalar component of the quaternion.
            q1 (float): X component of the quaternion.
            q2 (float): Y component of the quaternion.
            q3 (float): Z component of the quaternion.

        Returns:
            Rotation3D: Rotation constructed from the normalized quaternion.
        """
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
        """Rotation matrix about x-axis

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotation matrix around the x-axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        """Rotation matrix about y-axis

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotation matrix around the y-axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
     #Rotation matrix about z-axis
    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Rotation matrix about z-axis

        Args:
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotation matrix around the z-axis.
        """
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def apply(self, point: Point3D) -> Point3D:
        """Applies Rotation to a 3D Point so in effect the Rotation class acts upon the Point3D class

        Args:
            point (Point3D): Point to rotate.

        Returns:
            Point3D: Rotated point.
        """
        p_array = point.to_array()
        rotated = np.dot(self.matrix, p_array)
        return Point3D.from_array(rotated)
    
    def inverse(self) -> 'Rotation3D':
        """Applies Inverse Rotation (transpose of rotation matrix)

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            Rotation3D: Inverse rotation.
        """
        return Rotation3D(self.matrix.T)
    
    def compose(self, other: 'Rotation3D') -> 'Rotation3D':
        """Compose this rotation with another rotation. Note that this is particularly useful for kinematic chains

        Args:
            other (Rotation3D): Rotation to compose with this rotation.

        Returns:
            Rotation3D: Composition of the rotations.
        """
        return Rotation3D(np.dot(self.matrix, other.matrix))
    
    def determinant(self) -> float:
        """Get determinant of rotation matrix (should be +1). If negative that indicates a reflection is occurring

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            float: Determinant of the rotation matrix.
        """
        return np.linalg.det(self.matrix)
    
    def to_axis_angle(self) -> Tuple[Point3D, float]:
        """Convert rotation matrix to axis-angle representation

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            Tuple[Point3D, float]: Axis of rotation and rotation angle in radians.
        """
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
        """toString

        Args:
            None: This method uses the rotation matrix of the object.

        Returns:
            str: String representation of the rotation.
        """
        return f"Rotation3D(\n{self.matrix}\n)"

#3D Frame transformation class combining rotation and translation
class Frame3D:
    
    def __init__(self, rotation: Rotation3D = None, translation: Point3D = None):
        """Initializes the frame transformation with rotation and translation components.

        Args:
            rotation (Rotation3D): Rotation component of the frame.
            translation (Point3D): Translation component of the frame.

        Returns:
            None: This initializer configures the frame in place.
        """
        self.rotation = rotation if rotation is not None else Rotation3D()
        self.translation = translation if translation is not None else Point3D()
    
    @classmethod
    def identity(cls) -> 'Frame3D':
        """Create identity frame transformation

        Args:
            None: Uses default rotation and translation.

        Returns:
            Frame3D: Identity frame with no rotation and zero translation.
        """
        return cls()
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Frame3D':
        """Create frame from 4x4 homogeneous transformation matrix

        Args:
            matrix (np.ndarray): 4x4 homogeneous transformation matrix.

        Returns:
            Frame3D: Frame constructed from the provided matrix.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        
        rotation = Rotation3D(matrix[:3, :3])
        translation = Point3D(matrix[0, 3], matrix[1, 3], matrix[2, 3])
        return cls(rotation, translation)
    
    def apply(self, point: Point3D) -> Point3D:
        """Apply frame transformation to a 3D point

        Args:
            point (Point3D): Point to transform.

        Returns:
            Point3D: Transformed point.
        """
        rotated = self.rotation.apply(point)
        return rotated + self.translation
    
    def inverse(self) -> 'Frame3D':
        """Get inverse frame transformation

        Args:
            None: This method uses the frame's rotation and translation.

        Returns:
            Frame3D: Inverse of the frame transformation.
        """
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply(Point3D() - self.translation)
        return Frame3D(inv_rotation, inv_translation)
    
    def compose(self, other: 'Frame3D') -> 'Frame3D':
        """Compose this frame with another frame

        Args:
            other (Frame3D): Frame to compose with this frame.

        Returns:
            Frame3D: Composition of the two frames.
        """
        new_rotation = self.rotation.compose(other.rotation)
        new_translation = self.rotation.apply(other.translation) + self.translation
        return Frame3D(new_rotation, new_translation)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix

        Args:
            None: This method uses the frame's rotation and translation.

        Returns:
            np.ndarray: Homogeneous transformation matrix representation of the frame.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.matrix
        matrix[:3, 3] = [self.translation.x, self.translation.y, self.translation.z]
        return matrix
    
    def __repr__(self) -> str:
        """toString

        Args:
            None: This method uses the frame's rotation and translation.

        Returns:
            str: String representation of the frame.
        """
        return f"Frame3D(R={self.rotation}, t={self.translation})"

def compute_centroid(points: List[Point3D]) -> Point3D:
    """Compute centroid of a list of 3D points

    Args:
        points (List[Point3D]): Points used to compute the centroid.

    Returns:
        Point3D: Centroid of the provided points.
    """
    if not points:
        return Point3D()
    
    sum_x = sum(p.x for p in points)
    sum_y = sum(p.y for p in points)
    sum_z = sum(p.z for p in points)
    n = len(points)
    
    return Point3D(sum_x/n, sum_y/n, sum_z/n)

def compute_covariance_matrix(points_a: List[Point3D], points_b: List[Point3D]) -> np.ndarray:
    """Compute covariance matrix for point set registration

    Args:
        points_a (List[Point3D]): First set of points.
        points_b (List[Point3D]): Second set of points.

    Returns:
        np.ndarray: Covariance matrix derived from the two point sets.
    """
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
    """Create skew-symmetric matrix from 3D point

    Args:
        point (Point3D): Point from which to construct the matrix.

    Returns:
        np.ndarray: Skew-symmetric matrix representation of the point.
    """
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
