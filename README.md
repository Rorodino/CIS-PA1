# CIS PA1 - Electromagnetic Tracking System Calibration

Authors: Rohit Satisha and Sahana Raja

This project implements electromagnetic tracking system calibration algorithms including 3D point set registration, pivot calibration, and distortion calibration.

## Overview

The main problem tackled in this assignment was the development of a library of mathematical tools and subroutines relevant to computer-integrated surgery. This includes:

- 3D Cartesian mathematics package for points, rotations, and frame transformations
- Iterative Closest Point (ICP) algorithm for 3D point set registration
- Pivot calibration methods for EM and optical tracking systems
- Distortion calibration for electromagnetic tracking systems

## Installation

We recommend using [Anaconda3](https://www.anaconda.com/products/individual) to manage your environment.

### MacOS/Linux:
```bash
# Clone the repository
git clone <repository-url>
cd cis-pa1

# Create conda environment
conda env create -f environment.yml
conda activate cis-pa1

# Install the package
pip install -e .
```

### Windows:
Install Anaconda through the GUI, then use the commands above in a terminal.

## Usage

The main executable is `pa1.py` which provides access to all calibration functions:

### Problem 4a: Compute Fa frame transformations
```bash
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fa_a_registration
```

### Problem 4b: Compute Fd frame transformations
```bash
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --output_file Fd_a_registration
```

### Problem 5: EM pivot calibration
```bash
python pa1.py --name_3 pa1-debug-a-empivot --output_file A_EM_pivot
```

### Problem 6: Optical pivot calibration
```bash
python pa1.py --name pa1-debug-a-calbody --name_4 pa1-debug-a-optpivot --output_file A_Optpivot
```

### Problem 7: Complete output (NAME-OUTPUT-1.TXT format)
```bash
python pa1.py --name pa1-debug-a-calbody --name_2 pa1-debug-a-calreadings --name_3 pa1-debug-a-empivot --name_4 pa1-debug-a-optpivot --output_file pa1-debug-a-output1
```

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest -s

# Run specific test file
pytest -s tests/test_cis_pa1.py

# Run specific test function
pytest -s tests/test_cis_pa1.py::test_point3d_operations
```

## Project Structure

```
cis-pa1/
├── pa1.py                 # Main executable script
├── programs/             # Core algorithm modules
│   ├── __init__.py
│   ├── cis_math.py       # 3D mathematical operations
│   ├── icp_algorithm.py  # ICP registration algorithm
│   ├── data_readers.py   # Data file parsing
│   ├── pivot_calibration.py # Pivot calibration methods
│   ├── distortion_calibration.py # Distortion calibration
│   └── output_writer.py  # Output file generation
├── tests/                # Unit tests
│   └── test_cis_pa1.py
├── output/              # Generated output files
├── logging/             # Log files
├── setup.py             # Package setup
├── environment.yml      # Conda environment
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Dependencies

- Python 3.8+
- NumPy >= 1.21.0
- Click >= 8.0.0
- pytest >= 6.0.0

## Citations

- **NumPy Library**: Used for numerical computations and linear algebra operations
- **Click Library**: Command-line interface framework
- **Dr. Taylor's Course Materials**: Mathematical foundations for 3D transformations and calibration methods
- **Besl & McKay 1992**: ICP algorithm implementation reference
- **Rodrigues' Formula**: Rotation matrix computation from axis-angle representation

## Authors

- **Rohit Satisha**: Primary algorithm implementation, mathematical foundations, ICP algorithm development, pivot calibration methods, and comprehensive testing framework
- **Sahana Raja**: Data processing pipeline, file I/O operations, output formatting, debugging and validation, test case development, and integration testing
