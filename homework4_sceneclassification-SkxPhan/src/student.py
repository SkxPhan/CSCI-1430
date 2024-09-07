import numpy as np
from tqdm import tqdm
from model import SVM
from skimage import io, util, transform, feature
from sklearn.cluster import MiniBatchKMeans
import scipy as sc
from sklearn import svm

"""
READ FIRST: Relationship Between Functions

Functions in this file can be classified into 3 groups based on their roles:
Group 1: feature-extracting functions
        a) get_tiny_images:
            read in the images from the input paths and size down the images;
            the output tiny images are used as features
        b) get_bags_of_words:
            read in the images from the input paths and
            turn each of the images into a histogram of oriented gradients (hog);
            the output histograms are used as features
Group 2: supplementary function for get_bags_of_words (the second function in Group 1)
        build_vocabulary:
            read in the images from the input paths and build a vocabulary using the images using K-Means;
            the output vocabulary are fed into get_bags_of_words
            (Only need to run this function in main.py once)
Group 3: classification functions
        a) nearest_neighbor_classify
            implement nearest-neighbor classifier
        b) svm_classify
            implement many-versus-one linear SVM classifier

In main.py, we will run different combinations of functions in Group 1 and Group 3, e.g.
    i) get_tiny_images + nearest_neighbor_classify
    ii) get_bags_of_words + nearest_neighbor_classify
    iii) get_bags_of_words + svm_classify
    to perform scene classification.
We recommend to implement the functions in the following order:
    1) get_tiny_images, nearest_neighbor_classify, THEN run (i) to see the performance;
    2) get_bags_of_words, THEN run (ii) to see the performance.
    3) svm_classify, THEN run (iii) to see the performance.

Read main.py for more details.
"""


def get_tiny_images(image_paths, extra_credit=False):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2gray,
                         skimage.io.imread, np.reshape
    """

    # TODO: Implement this function!
    # One simply resizes each image to a small fixed resolution (we recommend 16x16).
    # It works slightly better if the tiny image is made to have zero mean and unit length.

    n = len(image_paths)
    d = 256
    vectorized_images = np.zeros((n, d))

    new_shape = (16, 16)

    for i in range(n):
        image = io.imread(image_paths[i])
        height, width = image.shape
        smallest = min(height, width)

        # Crop image first to get a square image
        image = util.crop(
            image, (height - smallest, width - smallest), copy=False
        )

        # Resize image and avoid aliasing
        image = transform.resize(image, new_shape, anti_aliasing=True)

        vectorized_image = image.reshape((1, d))

        vectorized_image = (
            vectorized_image - np.mean(vectorized_image)
        ) / np.std(vectorized_image)

        norm = np.linalg.norm(vectorized_image)

        vectorized_images[i, :] = vectorized_image / norm

    return vectorized_images


def extract_HOG_features(image_path):
    cells_per_block = (4, 4)
    pixels_per_cell = (4, 4)
    feature_vector = True

    image = io.imread(image_path)
    HOG_descriptor = feature.hog(
        image,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=feature_vector,
    )
    features = HOG_descriptor.reshape(
        -1, cells_per_block[0] * cells_per_block[1] * 9
    )

    return features


def build_vocabulary(image_paths, vocab_size, extra_credit=False):
    """
    This function samples HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    We will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).


    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    or sklearn.cluster.MiniBatchKMeans if sklearn.cluster.KMeans takes to long for you.
    Note that this can take A LONG TIME to complete (upwards of ten minutes
    for large numbers of features and large max_iter), so set the max_iter argument
    to something low (we used 100) and be patient. You may also find success setting
    the "tol" argument (see documentation for details).

    Once the vocabulary is created, it is saved as vocab.npy in a call in main.py.
    If you then use the flag `--load_vocab` on launch, it will load the vocab instead
    of recreating it. Hey presto!
    """

    # TODO: Implement this function!
    num_imgs = len(image_paths)
    images_features = []

    for i in tqdm(range(num_imgs), desc="Building Vocab"):
        images_features.append(extract_HOG_features(image_paths[i]))

    images_features = np.vstack(images_features)

    KMeans_iter = 100
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size, max_iter=KMeans_iter, verbose=0
    )
    print("Running KMeans...")
    kmeans.fit(images_features)

    return kmeans.cluster_centers_


def get_bags_of_words(image_paths, vocab, extra_credit=False):
    """
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    """

    # TODO: Implement this function!
    num_imgs = len(image_paths)
    vocab_size = len(vocab)
    bags_of_words = np.zeros((num_imgs, vocab_size))

    for i in tqdm(range(num_imgs), desc="Building bag of words for image"):
        features = extract_HOG_features(image_paths[i])

        # For each row of features, compute the closest word
        distances = sc.spatial.distance.cdist(features, vocab)
        closest_word_index = np.argsort(distances)[:, 0]

        # Build the histogram of words for the image
        bags_of_words[i, :] = np.bincount(
            closest_word_index, minlength=vocab_size
        )

    return bags_of_words


def svm_classify(
    train_image_feats, train_labels, test_image_feats, extra_credit=False
):
    """
    This function will predict a category for every test image by performing
    multi-class classification using a set of binary linear SVM classifiers
    trained on the training data. Then, these learned classifiers will be used
    on the test data.

    Inputs:
        train_image_feats:  An nxd numpy array, where n is the number of training
                            examples, and d is the image descriptor vector size.
        train_labels:       An n x 1 Python list containing the corresponding ground
                            truth labels for the training data.
        test_image_feats:   An m x d numpy array, where m is the number of test
                            images and d is the image descriptor vector size.

    Outputs:
        An m x 1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    """

    # TODO: Implement this function!

    # NOTES:
    # - We provide for you a binary linear support vector machine (SVM) classifier
    #   in `model.py`; the SVM class with function train().
    # - train() returns weights and biases; use these to score each image.
    # - You are not allowed to edit the SVM class. The autograder will use our own version of the SVM class.
    # - If you wish to implement more complex classifiers, use the extra_credit parameter.

    clf = svm.LinearSVC()
    clf.fit(train_image_feats, train_labels)

    return clf.predict(test_image_feats)


def nearest_neighbor_classify(
    train_image_feats, train_labels, test_image_feats, extra_credit=False
):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance. Play around
    with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    """

    k = 1

    # TODO:
    # 1) Find the k closest training features to each test image feature by some distance, e.g., Euclidean (L2)
    # for each row of test_image_feats, compute the Euclidean distance
    # with each row in train_image_feats and find the k closest one
    # How though? K-d tree? 256 features to compare!

    distances = sc.spatial.distance.cdist(test_image_feats, train_image_feats)
    k_nearest = np.argsort(distances)[:, :k]

    # 2) Determine the labels of those k features
    # We should ideally add a weighting factor w.r.t to the distance
    k_nearest_vote = sc.stats.mode(k_nearest, axis=1, keepdims=True)[0]

    # 3) Pick the most common label from the k
    k_nearest_label = np.take(train_labels, k_nearest_vote)

    # 4) Store that label in a list
    return k_nearest_label
