import numpy as np
import cv2
from feature_detection import computeHarrisValues, detectCorners, computeMOPSDescriptors, computeLocalMaximaHelper, produceMatches
import traceback
from scipy import ndimage


from PIL import Image

# Saving and loading cv2 points


def pickle_cv2(arr):
    index = []
    for point in arr:
        temp = (point[0], point[1], point[2], point[3])
        index.append(temp)
    return np.array(index)


def unpickle_cv2(arr):
    index = []
    for point in arr:
        temp = (point[0], point[1], point[2], point[3])
        index.append(temp)
    return np.array(index)

# Functions for testing elementwise correctness


def compare_array(arr1, arr2):
    return np.allclose(arr1, arr2, rtol=1e-3, atol=1e-3)


def compare_cv2_points(pnt1, pnt2):
    if not np.isclose(pnt1[0], pnt2[0], rtol=1e-3, atol=1e-5):
        return False
    if not np.isclose(pnt1[1], pnt2[1], rtol=1e-3, atol=1e-5):
        return False
    if not np.isclose(pnt1[2], pnt2[2], rtol=1e-3, atol=1e-5):
        return False
    if not np.isclose(pnt1[3], pnt2[3], rtol=1e-3, atol=1e-5):
        return False
    return True

# Testing function


def try_this(todo, run, truth, compare, *args, **kargs):
    '''
    Run a function, test the output with compare, and print and error if it doesn't work
    @arg todo (int or str): The Todo number
    @arg run (func): The function to run
    @arg truth (any): The correct output of the function 
    @arg compare (func->bool): Compares the output of the `run` function to truth and provides a boolean if correct
    @arg *args (any): Any arguments that should be passed to `run`
    @arg **kargs (any): Any kargs that should be passed to compare

    @return (int): The amount of things that failed
    '''
    print('Starting test for TODO {}'.format(todo))
    failed = 0
    try:
        output = run(*args)
    except Exception as e:
        traceback.print_exc()
        print("TODO {} threw an exception, see exception above".format(todo))
        return
    if type(output) is list or type(output) is tuple:
        for i in range(len(output)):
            if not compare(output[i], truth[i], **kargs):
                print("TODO {} doesn't pass test: {}".format(todo, i))
                failed += 1
    else:
        if not compare(output, truth, **kargs):
            print("TODO {} doesn't pass test".format(todo))
            failed += 1
    return failed


def compare_matches(match1, match2):
    return all(np.isclose(match1[idx], match2[idx], rtol=1e-3, atol=1e-5) for idx in range(3))


def test_produceMatches(todo, run, compare, image1, image2):
    print(f'Starting test for TODO {todo}')
    failed = 0
    try:
        # Generate descriptors for both images
        grayImage1 = cv2.cvtColor(image1.astype(
            np.float32)/255.0, cv2.COLOR_BGR2GRAY)
        grayImage2 = cv2.cvtColor(image2.astype(
            np.float32)/255.0, cv2.COLOR_BGR2GRAY)

        (harris1, orientation1) = computeHarrisValues(grayImage1)
        corners1 = detectCorners(harris1, orientation1)
        desc_img1 = computeMOPSDescriptors(image1, corners1)

        (harris2, orientation2) = computeHarrisValues(grayImage2)
        corners2 = detectCorners(harris2, orientation2)
        desc_img2 = computeMOPSDescriptors(image2, corners2)

        # Run the function to get matches
        output = run(desc_img1, desc_img2)
    except Exception as e:
        traceback.print_exc()
        print(f"TODO {todo} threw an exception, see exception above")
        return

    # Example of expected matches: This needs to be set up based on expected outcomes
    expected_matches = [(idx, idx, 0.8) for idx in range(
        min(len(desc_img1), len(desc_img2)))]  # Simplified expected result

    # Check matches
    if len(output) != len(expected_matches):
        print(
            f"TODO {todo} failed: number of matches incorrect, expected {len(expected_matches)}, got {len(output)}")
        failed += 1
    else:
        for i, match in enumerate(output):
            if not compare(match, expected_matches[i]):
                print(f"TODO {todo} doesn't pass test: {i}")
                failed += 1

    return failed


image = np.array(Image.open('resources/triangle1.jpg'))
grayImage = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)


def compute_and_save():
    (a, b) = computeHarrisValues(grayImage)  # Todo1
    c = computeLocalMaximaHelper(a)  # Todo2
    d = detectCorners(a, b)  # Todo3
    f = computeMOPSDescriptors(image, d)  # Todo4/5
    d_proc = pickle_cv2(d)
    np.savez('resources/arrays', a=a, b=b, c=c, d_proc=d_proc, f=f)
# Uncomment next line to overwrite test data (not recommended)
# compute_and_save()


'''
Load in the numpy arrays which hold results for triangle1.jpg.

These arrays can be accessed using loaded['<letter>']. For example, the
correct output for test 2 is 'c', so to see the correct output for test
2 you can inspect loaded['c'].  Important note: NumPy does not print
the entire array if it is very large --- you must print smaller pieces
(e.g., print repr(loaded['c'][0])).

If your tests fail you should inspect why it failed. In particular,
pay attention to the tolerances used by this testing script. It is
possible that your answer is correct but it barely falls outside the
tolerance range.

This is not the script used by the autograder. 
'''
loaded = np.load('resources/arrays.npz', allow_pickle=True)
d = unpickle_cv2(loaded['d_proc'])

try_this(1, computeHarrisValues, [loaded['a'],
         loaded['b']], compare_array, grayImage)

try_this(2, computeLocalMaximaHelper, loaded['c'], compare_array, loaded['a'])

try_this(3, detectCorners, d, compare_cv2_points, loaded['a'], loaded['b'])

try_this('4 and/or 5', computeMOPSDescriptors,
         loaded['f'], compare_array, image, d)


# Test for produceMatches (TODO 6)
# Load the original image
image = np.array(Image.open('resources/triangle1.jpg'))
# Create a modified version of the image (e.g., slight rotation)
modified_image = ndimage.rotate(image, 5, reshape=False)

# Integrating the test into the existing setup
test_produceMatches(6, produceMatches, compare_matches, image, modified_image)
