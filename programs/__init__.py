"""
CIS PA1 - Electromagnetic Tracking System Calibration
Authors: Rohit Satisha and Sahana Raja

This package contains the core modules for electromagnetic tracking system calibration.
"""

from .cis_math import Point3D, Rotation3D, Frame3D, compute_centroid
from .icp_algorithm import icp_algorithm, icp_with_known_correspondences, ICPResult
from .data_readers import (
    read_calbody_file, read_calreadings_file, 
    read_empivot_file, read_optpivot_file,
    CalibrationData, CalibrationReadings, EMPivotData, OptPivotData
)
from .pivot_calibration import em_pivot_calibration, opt_pivot_calibration
from .distortion_calibration import distortion_calibration, compute_fa_frame, compute_fd_frame
from .output_writer import (
    write_output1_file, write_transformation_matrix_file, 
    write_pivot_point_file, write_c_expected_file
)

__all__ = [
    'Point3D', 'Rotation3D', 'Frame3D', 'compute_centroid',
    'icp_algorithm', 'icp_with_known_correspondences', 'ICPResult',
    'read_calbody_file', 'read_calreadings_file', 'read_empivot_file', 'read_optpivot_file',
    'CalibrationData', 'CalibrationReadings', 'EMPivotData', 'OptPivotData',
    'em_pivot_calibration', 'opt_pivot_calibration',
    'distortion_calibration', 'compute_fa_frame', 'compute_fd_frame',
    'write_output1_file', 'write_transformation_matrix_file', 
    'write_pivot_point_file', 'write_c_expected_file'
]
