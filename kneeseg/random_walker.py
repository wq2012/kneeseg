import numpy as np
from skimage.segmentation import random_walker

def solve_random_walker(image, labels, beta=130, mode='cg_mg'):
    """
    Refines a segmentation using the Random Walker algorithm.
    
    Args:
        image: 3D numpy array (grayscale image).
        labels: 3D numpy array where:
            0: unlabeled voxels
            1, 2, ...: seed voxels for different classes
        beta: Weighting parameter for the Gaussian edge weight function.
        mode: Solver mode ('cg', 'cg_mg', 'bf').
        
    Returns:
        Refined label map.
    """
    # Normalize image to [0, 1] if not already
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_norm = (image - img_min) / (img_max - img_min)
    else:
        img_norm = image
        
    # Run random walker
    # Note: Skimage's RW is very robust.
    # spacing can be provided for anisotropic data
    
    # IMPORTANT: RW might remap labels if they are not contiguous.
    # We must handle the mapping manually to be safe.
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return labels # No seeds, nothing to do
        
    # Map to 1..K
    mapping = {label: i+1 for i, label in enumerate(unique_labels)}
    inverse_mapping = {i+1: label for i, label in enumerate(unique_labels)}
    
    mapped_labels = np.zeros_like(labels)
    for label, mapped_val in mapping.items():
        mapped_labels[labels == label] = mapped_val
        
    refined_mapped = random_walker(img_norm, mapped_labels, beta=beta, mode=mode)
    
    # Map back
    refined_labels = np.zeros_like(refined_mapped)
    for mapped_val, label in inverse_mapping.items():
        refined_labels[refined_mapped == mapped_val] = label
        
    return refined_labels

def cart_refinement_with_priors(image, prob_maps, masks, beta=130):
    """
    Custom refinement that combines RF probability maps with Random Walker.
    """
    # This is a placeholder for more advanced prior-based RW if needed.
    # For now, we take the argmax of prob_maps as the hard seeds where confidence is high.
    
    # Simple strategy: use high confidence RF predictions as seeds
    seeds = np.zeros(image.shape, dtype=int)
    for i in range(1, prob_maps.shape[0]): # Assuming 0 is background
        seeds[prob_maps[i] > 0.9] = i
        
    return solve_random_walker(image, seeds, beta=beta)
