import cv2
import numpy as np


def estimate_lane_lines(_main_image, _segmentation_mask):
    # Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_mask = np.zeros(_segmentation_mask.shape, dtype=np.uint8)
    # lane_mask[segmentation_output == 8] = [157, 234, 50]
    lane_mask[_segmentation_mask == 6] = 255

    # Perform Edge Detection
    edges = cv2.Canny(lane_mask, 50, 150, apertureSize=3)

    # Perform Line estimation
    lane_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=300)

    lane_lines = np.squeeze(lane_lines)

    # Filter out horizontal lines
    filtered_lines = []
    for line in lane_lines:
        x1, y1, x2, y2 = line
        if abs(y2 - y1) > abs(x2 - x1) * 0.4:
            filtered_lines.append(line)

    filtered_lines = np.squeeze(filtered_lines)

    vis_lanes(_main_image, filtered_lines)


def vis_lanes(image, lane_lines):
    for line in lane_lines:
        if line is not None:
            x1, y1, x2, y2 = line.astype(int)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)


def merge_lane_lines(lines):
    # Step 0: Define thresholds
    slope_similarity_threshold = 0.01
    intercept_similarity_threshold = 40

    # Step 1: Get slope and intercept of lines
    slopes, intercepts = get_slope_intercept(lines)

    clusters = []
    current_inds = []
    itr = 0

    # Step 3: Iterate over all remaining slopes and intercepts and cluster lines that are close to each other using a slope and intercept threshold.
    for slope, intercept in zip(slopes, intercepts):

        exists_in_clusters = np.array([itr in current for current in current_inds])

        if not exists_in_clusters.any():

            slope_cluster = np.logical_and(
                slopes < (slope + slope_similarity_threshold),
                slopes > (slope - slope_similarity_threshold))

            intercept_cluster = np.logical_and(
                intercepts < (intercept + intercept_similarity_threshold),
                intercepts > (intercept - intercept_similarity_threshold))

            inds = np.argwhere(slope_cluster & intercept_cluster)

            # if inds.size:
            #     current_inds.append(inds.flatten)
            #     clusters.append(lines[inds])

            if inds.size:
                current_inds.append(inds)
                cluster_lines = lines[inds]
                cluster_lines_mean = np.mean(cluster_lines, axis=0)
                cluster_lines_mean = np.expand_dims(cluster_lines_mean, axis=0)
                if len(current_inds) == 1:
                    merged_lines = cluster_lines_mean
                else:
                    merged_lines = np.concatenate((merged_lines, cluster_lines_mean), axis=0)

        itr += 1

    # Step 4: Merge all lines in clusters using mean averaging
    # merged_lines = [np.mean(cluster, axis=1) for cluster in clusters]
    merged_lines = np.squeeze(np.array(merged_lines), axis=1)

    return merged_lines


# def get_slope_intercept(lines):
#     slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + 0.001)
#     intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (lines[:, 2] + lines[:, 0])) / 2
#     return slopes, intercepts


def get_slope_intercept(lines):
    slopes = []
    intercepts = []
    for line in lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / ((x2 - x1) + 0.001)
        intercept = ((y2 + y1) - slope * (x2 - x1)) / 2
        slopes.append(slope)
        intercepts.append(intercept)
    return slopes, intercepts


def extrapolate_lines(lines, y_min, y_max):
    slopes, intercepts = get_slope_intercept(lines)

    new_lines = []

    for slope, intercept, in zip(slopes, intercepts):
        x1 = (y_min - intercept) / slope
        x2 = (y_max - intercept) / slope
        new_lines.append([x1, y_min, x2, y_max])

    return np.array(new_lines)

#
#
# def find_closest_lines(lines, point):
#     x0, y0 = point
#     distances = []
#     for line in lines:
#         x1, y1, x2, y2 = line
#         distances.append(((x2 - x1) * (y1 - y0) - (x1 - x0) *
#                           (y2 - y1)) / (np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)))
#
#     distances = np.abs(np.array(distances))
#     sorted = distances.argsort()
#
#     return lines[sorted[0:2], :]


# min_y = np.min(np.argwhere(road_mask == 1)[:, 0])
#
# extrapolated_lanes = extrapolate_lines(merged_lane_lines, WINDOW_HEIGHT, min_y)
# final_lanes = find_closest_lines(extrapolated_lanes, dataset_handler.lane_midpoint)
# plt.imshow(dataset_handler.vis_lanes(final_lanes))
