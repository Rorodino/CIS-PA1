This is the CIS PA1 Repository for Rohit Satisha and Sahana Raja


First I am going to break down the problem:

We have an electromagnetic tracking system that can measure the position of small markers relative to the base unit
  This measn that we cna get the points in the base unit coordinates

The key issue is that there is a distortion that occurs but we have a calibration object that we can use
  It has Nc markers
  We know that they have position ci in the frame of the calibration object where i obvioudsly represents each marker
    Thus Fc is the frame of the calibration object for which ci measures where the markers are with respect to that
    Ci is it in the frame of the em tracker base
  Similarly we have NA optical LED markers with position a_j in the frame of the calibration object
    A_j is the same LED in the frame of the optical tracker
  And we have Nd other markers with position d_j with respect to the tracker base
    D_j is the same trackers in the frame of the optical tracker

So in order to calibrate we are going to move the calibration object to a many places and so we will gather D, A, and C vectors for each relevant marker


Wer also have two posts:
  We have one with Nh LED markers with positions hi. Pivot Calibrations give lists of D and H
  We have one with Ng EM markers with positions gi. Pivot caliibrations give list of D and G





The main focus of assignment 1 is development of mathematical tools and subroutines that you may use in subsequent assignments. The specific goals are:

1. Develop (or develop proficiency with) a Cartesian math package for 3D points, rotations, and frame transformations.

2. Develop a 3D point set to 3D point set registration algorithm
We first havce to figure out what this means. Essentially, given a point cloud called A and a cpoint cloud called B we want to find the roattion and trasnaltion taht minimizes the sum of squared distances between corresponding points
Mathematically this woudl be:
  Sum across all i of (||Ra_i +t - bi||^2) where bi would be the point that corresponds to a_i to minimize this value
  However we do not know which b corresponds to which a a priori so we have to utilize an iterative approach to accomplish this which is the basis for the Iterated Closest Point Algorithm

So here is the algorithm that we will use:
  First we will staret with an initial guess of the Rotation and transaltion we need to use which by standard usually is R = I and t = 0.
  Now we use this to set initial correspoindences where we essentially can associate each point a to the closest point in b 

  Now after of course doign initial checks we need to apply the iterated closest poitn algorithm to find teh rotation and transaltion taht map cloud to cloud the best:
    1. We first compute the centroid of set a and set b
    2. Then we subtract the centroid of a froma ll points of a and centroid of b from all points in B
    3. Next we compute the covariance matrix which helps determien how A lines up with B. Note that if already aligned perfectlly the amtrix would look like a diagonal
    4. Next we compute the Single Value Decomposition of the covariance matrix. Mathematically, that is H= (U)(S)(V^T) whre these are all matrises and the U is direction in source space while V is         direction in target sace and S is the sclaras along the directions
        The wyas this works is that the points are broken down into 3 different matrcies where U is the directions of teh source points, V is for the target points and S is the scales
        So what this does is tell me the directions along which the axes are aligned but it does not tell me how much to roatte in order to achieve better alignement.
    5. Then using this information we compute the rotation which would be R = (V)(U^T). The U^T  part rotate sthe source points so that they are in the coordinate system of the SVD and the V rotates them so taht tehya re now in teh target axes. Note that if the det(R) is < 0 thenw e accidentally executed a reflection and we need to flip the last column of V and recmpute R which is just multiplying the last column by -1.

    6. Then the trasnaltion component is determined by how far apart the centroids are after roattion of the source centroids so centroid b - R * centroid A







3. Develop a “pivot” calibration method.

4. Given
a distortion calibration data set, as described above, compute the “expected” values


















