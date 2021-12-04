# Based on the paper "What Makes a Patch Distinct?" by Margolin et. al.
# Code written by Jake Oliger for Computer Vision B457 with Professor Reza at Indiana University

# SKImage for SLIC and PCA computation
from skimage.segmentation import slic, mark_boundaries
from sklearn.decomposition import PCA
from skimage.util import img_as_float
# For displaying segmentation
import matplotlib.pyplot as plt
# Because what project would be complete without Numpy and math
import numpy as np
import math
# For convenient image operations
from PIL import Image, ImageDraw, ImageOps
# For directory creation
import os

# There are some annoying warnings we want to ignore
np.warnings.filterwarnings('ignore')

###
# Returns salience of PIL Image `image` with display name `name`
# after scaling pixels to `scale`, using the given `segmentation_variance_cutoff`,
# and using patch sizes of `patch_size` as described in the Margolin et. al. paper.
# 
# Error checking on these inputs is sparse so be careful, lol.
# `name` should be a nonempty string
# `image` should be an image of size greater than the patch size at least, ideally much bigger
# `scale` should be a float greater than 0.1
# `segmentation_variance_cutoff` should be a float between 0.25 and 1.0
# `patch_size` should be an integer >= 1
def getSalience(name, image, scale, segmentation_variance_cutoff=0.25, patch_size=9):
    im_width, im_height = image.size

    image = image.resize((int(im_width * scale), int(im_height * scale)), resample=Image.NEAREST)
    image = image.resize((im_width, im_height), resample=Image.NEAREST)
    image = img_as_float(image)
    # Calculate the SLIC superpixels and prepare to gather data on them
    segments = np.array(slic(image, start_label=1, n_segments=100))
    num_segs = np.max(segments)
    seg_vals = [[] for _ in range(num_segs)]
    seg_color = np.zeros((num_segs, 3))

    im_width = image.shape[1]
    im_height = image.shape[0]

    # Per-pixel, per-metric salience
    salience_pattern = np.zeros((im_height, im_width))
    salience_color = np.zeros((im_height, im_width))

    # Aggregate the pixels of each superpixel into convenient arrays
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            try:
                seg_vals[segments[y][x] - 1].append([image[y][x][0], image[y][x][1], image[y][x][2]])
                seg_color[segments[y][x] - 1] = seg_color[segments[y][x] - 1] + np.array([image[y][x][0], image[y][x][1], image[y][x][2]])
            except IndexError:
                print(len(seg_vals))
                print(segments[y][x])
                print(image[y][x])
                raise IndexError

    # Find average color for each segment
    for n in range(num_segs):
        seg_color[n] = seg_color[n] / len(seg_vals[n])

    # Find color distance from each segment to all other segments, individually
    color_distance = np.zeros((num_segs, num_segs))
    for i in range(num_segs):
        for j in range(i + 1, num_segs):
            if i == j:
                continue
            cd = math.sqrt((seg_color[j][0] - seg_color[i][0])**2 + (seg_color[j][1] - seg_color[i][1])**2 + (seg_color[j][2] - seg_color[i][2])**2)
            color_distance[j][i] = cd
            color_distance[i][j] = cd

    # Find each segments distance to all other segments, cumulatively
    color_distance_by_seg = np.zeros((num_segs))
    max_color_distance = 0
    for i in range(num_segs):
        d = np.sum(color_distance[i])
        color_distance_by_seg[i] = d
        if d > max_color_distance:
            max_color_distance = d
    # Normalize the distances
    color_distance_by_seg = color_distance_by_seg / max_color_distance

    for x in range(im_width):
        for y in range(im_height):
            salience_color[y][x] = color_distance_by_seg[segments[y][x] - 1]

    # Determine 25% most variant SLIC superpixels so we can ignore the rest
    # (25% is default, this is controlled though by `segmentation_variance_cutoff`)
    seg_variances = np.ones((num_segs)) * -1
    for i in range(num_segs):
        px = np.array(seg_vals[i])
        var_red = np.var(px[:,0])
        var_green = np.var(px[:,1])
        var_blue = np.var(px[:,2])
        seg_variances[i] = var_red + var_green + var_blue
    n = int(num_segs * segmentation_variance_cutoff)
    ind = np.argpartition(seg_variances, -n)[-n:]

    average_patch = np.zeros((patch_size, patch_size, 3))
    num_averaged = 0

    patches = []

    # Get the average patch
    for x in range(0, im_width, patch_size):
        for y in range(0, im_height, patch_size):
            # Keep 'er movin' if this patch is split between segments or not in a high-variance segment
            if (segments[y][x] - 1) not in ind or y + (patch_size - 1) >= im_height or x + (patch_size - 1) >= im_width:
                continue
            this_seg = segments[y][x]
            if this_seg != segments[y + (patch_size - 1)][x]:
                continue
            if this_seg != segments[y][x + (patch_size - 1)]:
                continue
            if this_seg != segments[y + (patch_size - 1)][x + (patch_size - 1)]:
                continue
            patch = {'coords': (x, y), 'data': []}
            num_averaged += 1
            for i in range(patch_size):
                for j in range(patch_size):
                    p = image[y + j][x + i]
                    patch['data'].append([p[0], p[1], p[2]])
                    average_patch[j][i][0] += p[0]
                    average_patch[j][i][1] += p[1]
                    average_patch[j][i][2] += p[2]
            patches.append(patch)
    average_patch = average_patch / num_averaged
    average_patch_data = []
    for i in range(patch_size):
        for j in range(patch_size):
            p = average_patch[j][i]
            average_patch_data.append([p[0], p[1], p[2]])
    
    # Calculate PCA components for the average patch
    avg_pca = PCA()
    avg_pca.fit(average_patch_data)
    avg_components = avg_pca.components_
    avg_translated = np.zeros((patch_size, patch_size, 3))
    for i in range(patch_size):
        for j in range(patch_size):
            p = average_patch[j][i]
            # Save translated coordinates in PCA space
            avg_translated[j][i] = avg_components @ p

    distances = []

    for p in range(len(patches)):
        # Calculate PCA for each patch to be compared against the average
        pca = PCA()
        pca.fit(patches[p]['data'])
        this_components = pca.components_
        this_translated = np.zeros((patch_size, patch_size, 3))
        x = patches[p]['coords'][0]
        y = patches[p]['coords'][1]
        for i in range(patch_size):
            for j in range(patch_size):
                px = image[y + j][x + i]
                this_translated[j][i] = this_components @ px
        # Here's the money function, where we calculate distance in PCA space
        # to account for internal patch statistics for distinctness
        d = np.sum(np.power(avg_translated - this_translated, 2))
        distances.append(d)
        patches[p]['distance'] = d
    # Save maximum distance for normalization purposes
    maxd = np.max(np.array(distances))

    # Set the salience for each individual pixel
    for i in range(len(patches)):
        x0, y0 = (patches[i]['coords'][0], patches[i]['coords'][1])
        x1, y1 = (patches[i]['coords'][0] + patch_size, patches[i]['coords'][1] + patch_size)
        df = patches[i]['distance'] / maxd
        for i in range(x0, x1):
            for j in range(y0, y1):
                salience_pattern[j][i] = df

    # show the output of SLIC
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")
    plt.ioff()
    plt.savefig("debug/{}/segments_scale{:3.0f}.jpg".format(name, scale * 100))

    Image.fromarray(salience_pattern * 255).convert("RGB").save("debug/{}/pattern_salience_s{:3.0f}_p{:2.0f}.jpg".format(name, scale * 100, patch_size))
    Image.fromarray(salience_color * 255).convert("RGB").save("debug/{}/color_salience_s{:3.0f}_p{:2.0f}.jpg".format(name, scale * 100, patch_size))

    return (salience_pattern, salience_color)

# Basically wraps the getSalience function so we can calculate it repeatedly
# and conveniently with multiple parameters and average them all out
#
# This is called for in Margolin et. al. on page 1141 under implementation details.
def getMultifactorSalience(disp, img, scales, patch_sizes, seg_var_cut):
    n_scales = len(scales)
    n_patch_sizes = len(patch_sizes)
    n = n_scales * n_patch_sizes
    patterns = np.zeros((img.size[1], img.size[0]))
    colors = np.zeros((img.size[1], img.size[0]))
    for i in range(n_scales):
        for j in range(n_patch_sizes):
            p, c = getSalience(disp, img, scales[i], seg_var_cut, patch_sizes[j])
            patterns = patterns + p
            colors = colors + c
    patterns = patterns / n
    colors = colors / n

    Image.fromarray(patterns * 255).convert("RGB").save("debug/{}/pattern_salience_combined.jpg".format(disp))
    Image.fromarray(colors * 255).convert("RGB").save("debug/{}/color_salience_combined.jpg".format(disp))

    # This is the D(px) map described in the Margolin et. al. paper on pg 1143 
    return patterns * colors

# Some sample code for running this project
if __name__ == "__main__":
    filepath = "shepherd.jpg"
    display_name = filepath.replace(".jpg", "").replace(".png", "")
    if not os.path.exists("debug"):
        os.mkdir("debug")
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("output/" + display_name):
        os.mkdir("output/" + display_name)
    if not os.path.exists("debug/" + display_name):
        os.mkdir("debug/" + display_name)
    segmentation_variance_cutoff = 0.25
    patch_size = 3

    # load the image and convert it to a floating point data type
    img = Image.open("images/" + filepath)

    scales = [1.0, 0.5, 0.25]
    patch_sizes = [9, 6, 3]
    
    salience = getMultifactorSalience(display_name, img, scales, patch_sizes, 0.25)

    # Output the black/white salience image
    salience_img = Image.fromarray(salience * 255)
    salience_img.convert("RGB").save("debug/" + display_name + "/salience.jpg")
    salience_img.convert("RGB").save("output/" + display_name + "/salience.jpg")

    img = img.convert("RGB")

    # Output the composite of the salience and the original iamge
    composite = Image.composite(img, Image.new("RGB", salience_img.size), salience_img.convert("L"))
    composite.save("debug/" + display_name + "/salience-color.jpg")
    composite.save("output/" + display_name + "/salience-color.jpg")
