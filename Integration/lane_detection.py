import cv2
import numpy as np

from commons.CONSTANTS import Color


def estimate_lane_lines(_main_image, _segmentation_mask):
    # Create an image with pixels belonging to lane boundary categories from the output of semantic segmentation
    lane_mask = np.zeros(_segmentation_mask.shape, dtype=np.uint8)
    lane_mask[_segmentation_mask == 6] = 255

    # Perform Edge Detection
    edges = cv2.Canny(lane_mask, 50, 150, apertureSize=3)

    # Perform Line estimation
    lane_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=300)

    if lane_lines is not None:
        lane_lines = lane_lines.squeeze()
        # Filter out horizontal lines
        filtered_lines = __filter_lines(lane_lines, slope_threshold=0.4)
        merged_lines = merge_lane_lines(filtered_lines)
        vis_lanes(_main_image, merged_lines)


def __filter_lines(lines, slope_threshold=0.4):
    if lines.ndim > 2:
        lines = lines.squeeze()
    if lines.ndim == 1:
        lines = lines.reshape(-1, 4)
    assert lines.ndim == 2, f"Error: `lines` has {lines.ndim} dimensions should have 2 dimensions"
    assert lines.shape[1] == 4, f"Error: `lines` has {lines.shape[1]} columns should have 4 columns"
    return lines[np.abs(lines[:, 3] - lines[:, 1]) > slope_threshold * np.abs(lines[:, 2] - lines[:, 0])]


def vis_lanes(image, lane_lines):
    # Draw the lines on the image
    for line in lane_lines.astype(int):
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), Color.MAGENTA.value, 2)


def merge_lane_lines(lines, slope_sim_thresh=0.01, intercept_sim_thresh=40):
    slopes, intercepts = get_slope_intercept(lines)

    # Compute pairwise differences between slopes and intercepts
    slope_diffs = np.abs(slopes.reshape(-1, 1) - slopes)
    intercept_diffs = np.abs(intercepts.reshape(-1, 1) - intercepts)

    # Create a boolean mask for lines that are similar to each other
    similarity_mask = __calc_similarity_mask(slope_diffs, slope_sim_thresh, intercept_diffs, intercept_sim_thresh)

    # Loop over each component, and merge the lines in that component
    merged_lines = [np.mean(lines[similarity_mask[i]], axis=0) for i in range(len(lines))]
    return np.unique(merged_lines, axis=0, return_index=False)


def get_slope_intercept(lines):
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0] + np.finfo(float).eps)
    intercepts = ((lines[:, 3] + lines[:, 1]) - slopes * (lines[:, 2] + lines[:, 0])) / 2
    return slopes, intercepts


def __calc_similarity_mask(slope_diffs, slope_similarity_threshold, intercept_diffs, intercept_similarity_threshold):
    return np.logical_and(slope_diffs < slope_similarity_threshold, intercept_diffs < intercept_similarity_threshold)


def extrapolate_lines(lines, y_min, y_max):
    slopes, intercepts = get_slope_intercept(lines)
    x1 = (y_min - intercepts) / slopes
    x2 = (y_max - intercepts) / slopes
    new_lines = np.column_stack((x1, y_min * np.ones_like(x1), x2, y_max * np.ones_like(x2)))
    return new_lines
