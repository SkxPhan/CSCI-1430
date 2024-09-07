# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)

import csv
import sys
import argparse
import numpy as np
import scipy.io as scio

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

import student as student
import visualize
from helpers import cheat_feature_points, evaluate_correspondence


# This script
# (1) Loads and resizes images
# (2) Finds feature points in those images                 (you code this)
# (3) Visualizes feature points on these images            (you code this)
# (4) Describes each feature point with a local feature    (you code this)
# (5) Finds matching features                               (you code this)
# (6) Visualizes the matches
# (7) Evaluates the matches based on ground truth correspondences


def load_data(file_name):
    """
     1) Load stuff
     There are numerous other image sets in the supplementary data on the
     homework web page. You can simply download images off the Internet, as
     well. However, the evaluation function at the bottom of this script will
     only work for three particular image pairs (unless you add ground truth
     annotations for other image pairs). It is suggested that you only work
     with the two Notre Dame images until you are satisfied with your
     implementation and ready to test on additional images. A single scale
     pipeline works fine for these two images (and will give you full credit
     for this homework), but you will need local features at multiple scales to
     handle harder cases.

     If you want to add new images to test, replace the two images in the
     `custom` folder with your own image pairs. Make sure that the names match
     the ones in the elif for the custom folder. To run with your new images
     use python main.py -d custom.

    :param file_name: string for which image pair to compute correspondence for

        The first four strings can be used as shortcuts to the
        data files we give you

        1. notre_dame
        2. mt_rushmore
        3. e_gaudi
        4. custom

    :return: a tuple of the format (image1, image2, eval file)
    """

    # Note: these files default to notre dame, unless otherwise specified
    image1_file = "../data/NotreDame/NotreDame1.jpg"
    image2_file = "../data/NotreDame/NotreDame2.jpg"

    eval_file = "../data/NotreDame/NotreDameEval.mat"

    if file_name == "notre_dame":
        pass
    elif file_name == "mt_rushmore":
        image1_file = "../data/MountRushmore/Mount_Rushmore1.jpg"
        image2_file = "../data/MountRushmore/Mount_Rushmore2.jpg"
        eval_file = "../data/MountRushmore/MountRushmoreEval.mat"
    elif file_name == "e_gaudi":
        image1_file = "../data/EpiscopalGaudi/EGaudi_1.jpg"
        image2_file = "../data/EpiscopalGaudi/EGaudi_2.jpg"
        eval_file = "../data/EpiscopalGaudi/EGaudiEval.mat"
    elif file_name == "custom":
        image1_file = "../data/Custom/custom1.jpg"
        image2_file = "../data/Custom/custom2.jpg"
        eval_file = None

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2, eval_file


def main():
    """
    Reads in the data,

    Command line usage:

    python main.py -d | --data <image pair name> -p | --points <cheat or student points> [--sift]

    -d | --data - flag - required. specifies which image pair to match
    -p | --points - flag - required. specifies whether to use cheat points or student's feature points
    --sift - flag - optional. specifies whether to use the SIFT implementation in get_feature_descriptors()

    """

    # create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        choices=["notre_dame", "mt_rushmore", "e_gaudi", "custom"],
        help="Either notre_dame, mt_rushmore, e_gaudi, or custom. Specifies which image pair to match",
    )
    parser.add_argument(
        "-p",
        "--points",
        required=True,
        choices=["cheat_points", "student_points", "opencv_points"],
        help="Either cheat_points or student_points. Returns feature points for the image. Use \
                              cheat_points until get_feature_points() is implemented in student.py",
    )
    parser.add_argument(
        "--sift",
        required=False,
        default=False,
        action="store_true",
        help="Add this flag to use your SIFT implementation in get_feature_descriptors()",
    )
    args = parser.parse_args()

    # (1) Load in the data
    image1_color, image2_color, eval_file = load_data(args.data)

    # You don't have to work with grayscale images. Matching with color
    # information might be helpful. If you choose to work with RGB images, just
    # comment these two lines

    image1 = rgb2gray(image1_color)
    # Our own rgb2gray coefficients which match Rec.ITU-R BT.601-7 (NTSC) luminance conversion - only mino performance improvements and could be confusing to students
    # image1 = image1[:,:,0] * 0.2989 + image1[:,:,1] * 0.5870 + image1[:,:,2] * 0.1140
    image2 = rgb2gray(image2_color)
    # image2 = image2[:,:,0] * 0.2989 + image2[:,:,1] * 0.5870 + image2[:,:,2] * 0.1140

    # make images smaller to speed up the algorithm. This parameter
    # gets passed into the evaluation code, so don't resize the images
    # except for changing this parameter - We will evaluate your code using
    # scale_factor = 0.5, so be aware of this
    scale_factor = 0.5

    # Bilinear rescaling
    image1 = np.float32(rescale(image1, scale_factor))
    image2 = np.float32(rescale(image2, scale_factor))

    # width and height of each local feature, in pixels
    feature_width = 16

    # (2) Find distinctive points in each image. See Szeliski 4.1.1
    # !!! You will need to implement get_feature_points. !!!

    print("Getting feature points...")

    if args.points == "student_points":
        (x1, y1) = student.get_feature_points(image1, feature_width)
        (x2, y2) = student.get_feature_points(image2, feature_width)

    # For development and debugging get_feature_descriptors and match_features, you will likely
    # want to use the ta ground truth points, you can pass "-p cheat_points" to
    # the main function to do this. Note that the ground truth points for mt. rushmore
    # will not produce good results, so you'll have to use your own function for
    # that image pair.
    elif args.points == "cheat_points":
        (x1, y1, x2, y2) = cheat_feature_points(eval_file, scale_factor)

    elif args.points == "opencv_points":
        (x1, y1) = student.get_opencv_feature_points(image1)
        (x2, y2) = student.get_opencv_feature_points(image2)

    # Viewing your feature points on your images.
    # !!! You will need to implement plot_feature_points. !!!
    print("Number of feature points found (image 1):", len(x1))
    student.plot_feature_points(image1, x1, y1)
    print("Number of feature points found (image 2):", len(x2))
    student.plot_feature_points(image2, x2, y2)

    print("Done!")

    # 3) Create feature vectors at each feature point. Szeliski 4.1.2
    # !!! You will need to implement get_feature_descriptors. !!!

    print("Getting features...")

    if args.sift:
        mode = "sift"
    else:
        mode = "patch"

    image1_features = student.get_feature_descriptors(
        image1, x1, y1, feature_width, mode
    )
    image2_features = student.get_feature_descriptors(
        image2, x2, y2, feature_width, mode
    )

    print("Done!")

    # 4) Match features. Szeliski 4.1.3
    # !!! You will need to implement match_features !!!

    print("Matching features...")

    matches = student.match_features(image1_features, image2_features)

    print("Done!")

    # 5) Evaluation and visualization

    # The last thing to do is to check how your code performs on the image pairs
    # we've provided. The evaluate_correspondence function below will print out
    # the accuracy of your feature matching for your 50 most confident matches,
    # 100 most confident matches, and all your matches. It will then visualize
    # the matches by drawing green lines between points for correct matches and
    # red lines for incorrect matches. The visualizer will show the top
    # num_pts_to_visualize most confident matches, so feel free to change the
    # parameter to whatever you like.

    print("Matches: " + str(matches.shape[0]))

    if args.data == "custom":
        print("Visualizing on custom images...")
        visualize.show_correspondences_custom_image(
            image1_color,
            image2_color,
            x1,
            y1,
            x2,
            y2,
            matches,
            scale_factor,
            args.data + "_matches.png",
        )
    else:
        evaluate_correspondence(
            image1_color,
            image2_color,
            eval_file,
            scale_factor,
            x1,
            y1,
            x2,
            y2,
            matches,
            args.data + "_matches.png",
        )

    return


if __name__ == "__main__":
    main()
