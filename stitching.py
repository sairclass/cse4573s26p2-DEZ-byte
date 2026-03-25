'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Anaconda OpenMP kept crashing on my Windows laptop.
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# Grouped the repeated feature detection and homography stuff into helpers
# so I don't have to write it twice for task 1 and 2.

def _extract_and_match(img1, img2, num_features=2000):
    """Run SIFT and get matched points using MNN."""
    gray1 = K.color.rgb_to_grayscale(img1)
    gray2 = K.color.rgb_to_grayscale(img2)
    sift = K.feature.SIFTFeature(num_features=num_features, device=img1.device)
    with torch.no_grad():
        lafs1, _, descs1 = sift(gray1)
        lafs2, _, descs2 = sift(gray2)
        # Mutual nearest neighbor filters out a lot of the bad matches
        dists, idxs = K.feature.match_mnn(descs1[0], descs2[0])
        pts1 = K.feature.get_laf_center(lafs1)[0, idxs[:, 0]]
        pts2 = K.feature.get_laf_center(lafs2)[0, idxs[:, 1]]
    return pts1, pts2

def _compute_homography(pts_src, pts_dst, inl_th=3.0):
    """Finds the homography matrix using RANSAC."""
    ransac = K.geometry.ransac.RANSAC(model_type='homography', inl_th=inl_th)
    H, inliers = ransac(pts_src, pts_dst)
    # RANSAC sometimes drops the batch dimension, so we add it back
    if H.dim() == 2:
        H = H.unsqueeze(0)
    return H

def _compute_canvas(H_2to1, H1, W1, device):
    """Calculates the new canvas size after warping and creates a translation matrix to prevent negative coordinates."""
    corners1 = torch.tensor([[[0., 0.],[W1, 0.], [W1, H1], [0., H1]]], device=device)
    corners2 = corners1.clone()
    corners2_warped = K.geometry.transform_points(H_2to1, corners2)
    all_corners = torch.cat([corners1, corners2_warped], dim=1)
    min_xy = all_corners.min(dim=1)[0][0]
    max_xy = all_corners.max(dim=1)[0][0]
    min_x, min_y = int(min_xy[0].floor().item()), int(min_xy[1].floor().item())
    max_x, max_y = int(max_xy[0].ceil().item()), int(max_xy[1].ceil().item())
    T = torch.eye(3, device=device).unsqueeze(0)
    T[0, 0, 2] = -min_x
    T[0, 1, 2] = -min_y
    return T, max_x - min_x, max_y - min_y

def _center_weight_mask(h, w, device):
    """Creates a mask that's 1 in the center and fades to 0 at the edges. 
    This helps blend overlapping images without harsh seams."""
    # Horizontal fade
    x = torch.linspace(0, 1, w, device=device)
    wx = torch.min(x, 1.0 - x)
    # Vertical fade
    y = torch.linspace(0, 1, h, device=device)
    wy = torch.min(y, 1.0 - y)
    # Combine them into a 2D mask
    weight = wy.unsqueeze(1) * wx.unsqueeze(0)  
    return weight.unsqueeze(0).unsqueeze(0)  


#  Task 1 
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img_list = list(imgs.values())
    # Convert to float for Kornia
    img1, img2 = img_list[0].float() / 255.0, img_list[1].float() / 255.0

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape
    device = img1.device

    # Find matches and calculate the warp from img2 to img1
    pts1, pts2 = _extract_and_match(img1, img2)
    H_2to1 = _compute_homography(pts2, pts1)

    # Calculate the final canvas size
    T, out_W, out_H = _compute_canvas(H_2to1, H, W, device)

    H_img1 = T           # Shift img1 into the canvas
    H_img2 = T @ H_2to1  # Warp img2 to match img1, then shift it

    warped1 = K.geometry.transform.warp_perspective(img1, H_img1, dsize=(out_H, out_W))
    warped2 = K.geometry.transform.warp_perspective(img2, H_img2, dsize=(out_H, out_W))

    # Create masks to track where the actual image pixels end up
    mask_ones = torch.ones((B, 1, H, W), device=device)
    valid1 = (K.geometry.transform.warp_perspective(mask_ones, H_img1, dsize=(out_H, out_W)) > 0.5).float()
    valid2 = (K.geometry.transform.warp_perspective(mask_ones, H_img2, dsize=(out_H, out_W)) > 0.5).float()

    overlap = valid1 * valid2
    only1 = valid1 * (1.0 - valid2)   # Pixels exclusive to img1
    only2 = valid2 * (1.0 - valid1)   # Pixels exclusive to img2

    # Foreground removal, find where the images differ (meaning someone moved). 
    # Then figure out which one is the clean background and use that.

    # Average color difference
    diff = (warped1 - warped2).abs().mean(dim=1, keepdim=True)
    
    # 0.08 threshold worked best. 
    fg_mask = (diff > 0.08).float()

    # Dilate 
    kernel = torch.ones(7, 7, device=device)
    fg_mask = K.morphology.dilation(fg_mask, kernel)
    
    fg_mask = K.filters.gaussian_blur2d(fg_mask, (15, 15), (4.0, 4.0))
    fg_mask = fg_mask.clamp(0, 1)

    # Create a fake background reference by heavily blurring the clean overlapping areas.
    clean_w = overlap * (1.0 - fg_mask)  # Only use clean pixels
    bg_sum = K.filters.gaussian_blur2d(
        (warped1 + warped2) / 2.0 * clean_w, (51, 51), (15.0, 15.0)
    )
    weight_sum = K.filters.gaussian_blur2d(clean_w, (51, 51), (15.0, 15.0))
    bg_ref = bg_sum / (weight_sum + 1e-6) 

    # Compare both images to the fake background. The one that's closer is the clean one.
    dist1 = (warped1 - bg_ref).abs().mean(dim=1, keepdim=True)
    dist2 = (warped2 - bg_ref).abs().mean(dim=1, keepdim=True)
    
    # Blur the distances slightly so the decision boundary isn't noisy
    dist1 = K.filters.gaussian_blur2d(dist1, (9, 9), (3.0, 3.0))
    dist2 = K.filters.gaussian_blur2d(dist2, (9, 9), (3.0, 3.0))
    use_img1 = (dist1 < dist2).float()  # Pick the cleaner image

    # Average the clean parts, but use our selected clean pixels for the foreground areas.
    clean_blend = (warped1 + warped2) / 2.0
    fg_blend = warped1 * use_img1 + warped2 * (1.0 - use_img1)
    overlap_result = clean_blend * (1.0 - fg_mask) + fg_blend * fg_mask

    img = warped1 * only1 + warped2 * only2 + overlap * overlap_result

    if img.shape[0] == 1:
        img = img.squeeze(0)

    # Convert back to uint8 for saving
    img = (img * 255.0).clamp(0, 255).byte()
    return img


# Task 2 
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: torch.Tensor of the output image.
    """
    img_keys = list(imgs.keys())
    N = len(img_keys)

    # Convert to float and add batch dimension
    processed = []
    for k in img_keys:
        t = imgs[k].float() / 255.0
        if t.dim() == 3:
            t = t.unsqueeze(0)
        processed.append(t)

    device = processed[0].device
    _, C, H, W = processed[0].shape

    # Build the overlap matrix 
    # Checking every pair since the panorama might wrap around.
    overlap_matrix = torch.zeros((N, N), dtype=torch.int32, device=device)
    pairwise_H = {}  # Cache the homographies so we don't recalculate them later
    mask_ones = torch.ones((1, 1, H, W), device=device)

    for i in range(N):
        overlap_matrix[i, i] = 1  # Images always overlap with themselves
        for j in range(i + 1, N):
            try:
                pts_i, pts_j = _extract_and_match(processed[i], processed[j])
                
                # Skip if there aren't enough matches
                if len(pts_i) < 15:
                    continue

                H_j_to_i = _compute_homography(pts_j, pts_i)

                # 20% overlap - warp a mask and calculate the intersection area
                warped_mask = K.geometry.transform.warp_perspective(
                    mask_ones, H_j_to_i, dsize=(H, W)
                )
                intersection = (warped_mask * mask_ones).sum()
                area = mask_ones.sum()
                
                if (intersection / area) >= 0.20:
                    overlap_matrix[i, j] = 1
                    overlap_matrix[j, i] = 1
                    pairwise_H[(j, i)] = H_j_to_i
                    
                    # Save the reverse direction for the BFS later
                    H_i_to_j = _compute_homography(pts_i, pts_j)
                    pairwise_H[(i, j)] = H_i_to_j
            except Exception:
                continue

    # Find the anchor image, take the one with the most overlaps, keeps distortion down.
    overlap_counts = overlap_matrix.sum(dim=1)
    ref = int(overlap_counts.argmax().item())

    # BFS to link everything back to the anchor 
    H_to_ref = {ref: torch.eye(3, device=device).unsqueeze(0)}
    visited = {ref}
    queue =[ref]

    while queue:
        cur = queue.pop(0)
        for nxt in range(N):
            if nxt in visited or overlap_matrix[cur, nxt] == 0:
                continue
            if (nxt, cur) in pairwise_H:
                H_to_ref[nxt] = H_to_ref[cur] @ pairwise_H[(nxt, cur)]
                visited.add(nxt)
                queue.append(nxt)

    # Calculate the final canvas size by warping all corners
    all_corners =[]
    for idx, H_r in H_to_ref.items():
        _, _, hi, wi = processed[idx].shape
        corners = torch.tensor([[[0., 0.], [wi, 0.], [wi, hi],[0., hi]]], device=device)
        warped_c = K.geometry.transform_points(H_r, corners)
        all_corners.append(warped_c)

    all_corners = torch.cat(all_corners, dim=1)
    min_xy = all_corners.min(dim=1)[0][0]
    max_xy = all_corners.max(dim=1)[0][0]
    min_x = int(min_xy[0].floor().item())
    min_y = int(min_xy[1].floor().item())
    max_x = int(max_xy[0].ceil().item())
    max_y = int(max_xy[1].ceil().item())
    canvas_w, canvas_h = max_x - min_x, max_y - min_y

    # Shift everything
    T = torch.eye(3, device=device).unsqueeze(0)
    T[0, 0, 2] = -min_x
    T[0, 1, 2] = -min_y

    # Using a center-weighted mask so the edges fade out smoothly.
    canvas = torch.zeros((1, C, canvas_h, canvas_w), device=device)
    weight_canvas = torch.zeros((1, 1, canvas_h, canvas_w), device=device)

    for idx, H_r in H_to_ref.items():
        _, _, hi, wi = processed[idx].shape
        H_final = T @ H_r

        # Get the fade mask for this image
        center_w = _center_weight_mask(hi, wi, device)

        warped_img = K.geometry.transform.warp_perspective(
            processed[idx], H_final, dsize=(canvas_h, canvas_w)
        )
        warped_weight = K.geometry.transform.warp_perspective(
            center_w, H_final, dsize=(canvas_h, canvas_w)
        )

        canvas += warped_img * warped_weight
        weight_canvas += warped_weight

    # Average out the overlapping pixels
    final = canvas / (weight_canvas + 1e-7)

    # Crop out the extra black space
    nz = torch.nonzero(weight_canvas[0, 0] > 0.01)
    if len(nz) > 0:
        y_min, x_min = nz.min(dim=0)[0]
        y_max, x_max = nz.max(dim=0)[0]
        final = final[:, :, y_min:y_max + 1, x_min:x_max + 1]

    if final.shape[0] == 1:
        final = final.squeeze(0)

    final = (final * 255.0).clamp(0, 255).byte()
    return final, overlap_matrix
