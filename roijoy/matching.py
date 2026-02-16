"""Feature-based ROI matching across images using ORB."""
import numpy as np
import cv2


def match_roi(source_rgb: np.ndarray, target_rgb: np.ndarray,
              vertices: list[tuple[float, float]],
              min_inliers: int = 10) -> dict:
    """Match an ROI from source image to target image using ORB features.

    Args:
        source_rgb: Source image (uint8, H x W x 3)
        target_rgb: Target image (uint8, H x W x 3)
        vertices: Polygon vertices in source image (x, y) pixel coords
        min_inliers: Minimum RANSAC inliers to consider match valid

    Returns:
        Dict with: success (bool), vertices (transformed), method (str),
                   n_inliers (int), confidence (float)
    """
    src_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return copy_roi(vertices)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_inliers:
        return copy_roi(vertices)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return copy_roi(vertices)

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < min_inliers:
        return copy_roi(vertices)

    pts = np.float32(vertices).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    return {
        "success": True,
        "vertices": [(float(p[0]), float(p[1])) for p in transformed],
        "method": "feature",
        "n_inliers": n_inliers,
        "confidence": n_inliers / len(good_matches),
    }


def copy_roi(vertices: list[tuple[float, float]]) -> dict:
    """Fallback: copy polygon vertices as-is to target image."""
    return {
        "success": True,
        "vertices": list(vertices),
        "method": "copy",
        "n_inliers": 0,
        "confidence": 0.0,
    }
