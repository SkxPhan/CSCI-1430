import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def compute_sampson_distance(F, pts1, pts2):
    """
    Compute the Sampson distance for each point correspondence.

    Parameters:
    F (numpy.ndarray): The 3x3 fundamental matrix.
    pts1 (numpy.ndarray): An Nx3 array of homogeneous coordinates in the first image.
    pts2 (numpy.ndarray): An Nx3 array of homogeneous coordinates in the second image.

    Returns:
    numpy.ndarray: An array of Sampson distances for each point correspondence.
    """
    # Compute the epipolar lines in the second image for points in the first image
    pts1 = np.column_stack((pts1, np.ones(pts1.shape[0])))
    pts2 = np.column_stack((pts2, np.ones(pts2.shape[0])))

    Fx1 = F @ pts1.T
    Fx1 = Fx1.T  # Transpose to match the shape of pts1

    # Compute the epipolar lines in the first image for points in the second image
    Ftx2 = F.T @ pts2.T
    Ftx2 = Ftx2.T  # Transpose to match the shape of pts2

    # Compute the numerator of the Sampson distance
    numerator = (np.sum(pts2 * (F @ pts1.T).T, axis=1)) ** 2

    # Compute the denominator of the Sampson distance
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2

    # Compute the Sampson distance
    sampson_distance = numerator / denom

    return sampson_distance.reshape((-1, 1))


def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points

    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
             residual, the error in the estimation of M given the point sets
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters
    )
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    # # Placeholder values. This M matrix came from a call to rand(3,4). It leads to a high residual.
    # print("Randomly setting matrix entries as a placeholder")
    # M = np.array(
    #     [
    #         [0.1768, 0.7018, 0.7948, 0.4613],
    #         [0.6750, 0.3152, 0.1136, 0.0480],
    #         [0.1020, 0.1725, 0.7244, 0.9932],
    #     ]
    # )
    # residual = 7  # Arbitrary stencil code initial value placeholder

    print("Estimating the projection matrix")

    # 1) Vectorize points2d into a single vector (b)
    b = points2d.reshape((-1, 1))

    # 2) Build the matrix A using points2d and points3d (A)
    A = points3d.repeat(repeats=2, axis=0) * -b

    # Construct a vector of 1 (same len as points3d) and matrix of 0 ((len points3d, 4 x 0))
    points3d_homogeneous = np.column_stack(
        (points3d, np.ones(points3d.shape[0]))
    )

    right = np.hstack(
        (
            points3d_homogeneous,
            np.zeros((points3d.shape[0], 8)),
            points3d_homogeneous,
        )
    ).reshape((-1, 8))

    A = np.column_stack((right, A))

    # 3) Use numpy.linalg.lstsq to obtain the parameters of the matrix M (x)
    x, residual = np.linalg.lstsq(A, b, rcond=None)[:2]

    # 4) Reshape the vector x into a 3x4 matrix
    M = np.append(x, 1)
    M = M.reshape((3, 4))

    return M, residual


def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices.

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T


def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will transform a point into
    a line within the second image - the epipolar line - such that F x' = l.
    Fitting a fundamental matrix to a set of points will try to minimize the
    error of all points x to their respective epipolar lines transformed
    from x'. The residual can be computed as the difference from the known
    geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the error in the estimation
    """
    ########################
    # TODO: Your code here #
    ########################

    # Arbitrary intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0, 0, -0.0004], [0, 0, 0.0032], [0, -0.0044, 0.1034]])
    # residual = 5  # Arbitrary stencil code initial value placeholder

    A = np.column_stack(
        (
            points1[:, 0][:, np.newaxis] * points2,
            points1[:, 0],
            points1[:, 1][:, np.newaxis] * points2,
            points1[:, 1],
            points2,
            np.ones((points1.shape[0], 1)),
        )
    )

    b = np.zeros((A.shape[0], 1))

    # F_matrix, residual = np.linalg.lstsq(A, b, rcond=None)[:2]

    # Compute SVD of A
    U, S, VT = np.linalg.svd(A)

    # # Compute pseudo-inverse of S
    # S_inv = np.linalg.inv(np.diag(S))

    # # Compute least squares solution using SVD
    # F_matrix = VT.T @ S_inv @ U.T @ b

    x = VT[-1, :]
    F_matrix = x.reshape((3, 3))

    # Resolve det(F) = 0 constraint using SVD
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    # Compute residual
    residual = np.linalg.norm(A @ F_matrix.reshape((-1, 1)) - b)

    F_matrix = x.reshape((3, 3))

    return F_matrix, residual


def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points.

    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the error induced by best_Fmatrix

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)

    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to your 'estimate_fundamental_matrix()'

    # For your report, we ask you to visualize RANSAC's
    # convergence over iterations. (residual of best matrix over iter)
    # For each iteration, append your inlier count and residual to the global variables:
    #   inlier_counts = []
    #   inlier_residuals = []
    # Then add flag --visualize-ransac to plot these using visualize_ransac()

    # 1) Select randomly 8 points in matches1 and corresponding matches in matches2
    # A = np.column_stack(
    #     (
    #         matches1[:, 0][:, np.newaxis] * matches2,
    #         matches1[:, 0],
    #         matches1[:, 1][:, np.newaxis] * matches2,
    #         matches1[:, 1],
    #         matches2,
    #         np.ones((matches1.shape[0], 1)),
    #     )
    # )

    n_sample = 8
    RANSAC_iter = 2000
    # residual_threshold = 1e-3
    residual_threshold = 10
    best_inlier_count = 0
    best_inlier_residual = 1e10

    for i in range(RANSAC_iter):
        random_indexes = np.random.randint(matches1.shape[0], size=n_sample)

        sample_points1 = matches1[random_indexes]
        sample_points2 = matches2[random_indexes]

        # 2) From thoses pairs, compute the Fmatrix using OpenCV implementation
        # Fmatrix, _ = cv2.findFundamentalMat(
        #     sample_points1, sample_points2, cv2.FM_8POINT, 1e10, 0, 1
        # )

        Fmatrix, _ = estimate_fundamental_matrix(sample_points1, sample_points2)

        if Fmatrix is None:
            continue

        # 3) Count the number of inlier and the residual value x.T * F * x' = 0
        # residuals = np.square(A @ Fmatrix.reshape((-1, 1)))
        residuals = compute_sampson_distance(Fmatrix, matches1, matches2)
        mask = np.where(residuals < residual_threshold)[0]

        inlier_count = np.count_nonzero(mask)
        inlier_counts.append(inlier_count)

        inlier_residual = np.sum(residuals[mask])
        inlier_residuals.append(inlier_residual)

        # Best model is the one with the lowest residual
        if best_inlier_count < inlier_count:
            best_inlier_count = inlier_count
            best_inlier_residual = inlier_residual
            best_Fmatrix = Fmatrix

            best_inliers_a = matches1[mask]
            best_inliers_b = matches2[mask]

    return best_Fmatrix, best_inliers_a, best_inliers_b, best_inlier_residual


def matches_to_3d(points2d_1, points2d_2, M1, M2, threshold=1.0):
    """
    Given two sets of corresponding 2D points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq().

    You may find that some 3D points have high residual/error, in which case you
    can return a subset of the 3D points that lie within a certain threshold.
    In this case, also return subsets of the initial points2d_1, points2d_2 that
    correspond to this new inlier set. You may modify the default value of threshold above.
    All local helper code that calls this function will use this default value, but we
    will pass in a different value when autograding.

    N is the input number of point correspondences
    M is the output number of 3D points / inlier point correspondences; M could equal N.

    :param points2d_1: [N x 2] points from image1
    :param points2d_2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :param threshold: scalar value representing the maximum allowed residual for a solved 3D point

    :return points3d_inlier: [M x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points2d_1 and points2d_2
    :return points2d_1_inlier: [M x 2] points as subset of inlier points from points2d_1
    :return points2d_2_inlier: [M x 2] points as subset of inlier points from points2d_2
    """
    ########################
    # TODO: Your code here #

    # Initial random values for 3D points
    # points3d_inlier = np.random.rand(len(points2d_1), 3)
    # points2d_1_inlier = np.array(
    #     points2d_1, copy=True
    # )  # only modify if using threshold
    # points2d_2_inlier = np.array(
    #     points2d_2, copy=True
    # )  # only modify if using threshold

    points3d_inlier = []
    points2d_1_inlier = []
    points2d_2_inlier = []

    # Solve for ground truth points
    for p, q in zip(points2d_1, points2d_2):
        u1, v1 = p
        u2, v2 = q

        A = np.array(
            [
                [
                    u1 * M1[2, 0] - M1[0, 0],
                    u1 * M1[2, 1] - M1[0, 1],
                    u1 * M1[2, 2] - M1[0, 2],
                ],
                [
                    v1 * M1[2, 0] - M1[1, 0],
                    v1 * M1[2, 1] - M1[1, 1],
                    v1 * M1[2, 2] - M1[1, 2],
                ],
                [
                    u2 * M2[2, 0] - M2[0, 0],
                    u2 * M2[2, 1] - M2[0, 1],
                    u2 * M2[2, 2] - M2[0, 2],
                ],
                [
                    v2 * M2[2, 0] - M2[1, 0],
                    v2 * M2[2, 1] - M2[1, 1],
                    v2 * M2[2, 2] - M2[1, 2],
                ],
            ]
        )

        b = np.array(
            [
                [M1[0, 3] - u1 * M1[2, 3]],
                [M1[1, 3] - v1 * M1[2, 3]],
                [M2[0, 3] - u2 * M2[2, 3]],
                [M2[1, 3] - v2 * M2[2, 3]],
            ]
        )

        x, residual = np.linalg.lstsq(A, b, rcond=None)[:2]

        if residual < threshold:
            points3d_inlier.append(x)
            points2d_1_inlier.append(p)
            points2d_2_inlier.append(q)

    points3d_inlier = np.asarray(points3d_inlier).squeeze()
    points2d_1_inlier = np.asarray(points2d_1_inlier)
    points2d_2_inlier = np.asarray(points2d_2_inlier)

    ########################

    return points3d_inlier, points2d_1_inlier, points2d_2_inlier


# /////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []


def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    plt.figure(1, figsize=(8, 8))
    plt.subplot(211)
    plt.plot(
        iterations, inlier_counts, label="Current Inlier Count", color="red"
    )
    plt.plot(
        iterations, best_inlier_counts, label="Best Inlier Count", color="blue"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title("Current Inliers vs. Best Inliers per Iteration")
    plt.legend()

    plt.subplot(212)
    plt.plot(
        iterations,
        inlier_residuals,
        label="Current Inlier Residual",
        color="red",
    )
    plt.plot(
        iterations,
        best_inlier_residuals,
        label="Best Inlier Residual",
        color="blue",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Current Residual vs. Best Residual per Iteration")
    plt.legend()
    plt.show()
