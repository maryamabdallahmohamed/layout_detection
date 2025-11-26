import numpy as np

def sort_single_column(bboxes):
    """
    Orders bounding boxes for single column layout (top to bottom).
    """
    if not bboxes:
        return []
    # Sort by y-coordinate (top to bottom)
    return sorted(bboxes, key=lambda b: b.bbox[1])

def sort_two_column_rtl(bboxes):
    """
    Groups bounding boxes into two columns (right-to-left order) for Arabic documents.
    Returns: right column (top->bottom) -> left column (top->bottom).
    """
    if not bboxes:
        return []

    # Compute horizontal centers
    centers = [((b.bbox[0] + b.bbox[2]) / 2, b) for b in bboxes]
    centers.sort(key=lambda x: x[0])  # sort by x-center (left -> right)
    xs = [c[0] for c in centers]

    # Find the largest horizontal gap -> column separator
    if len(xs) > 1:
        diffs = np.diff(xs)
        split_idx = np.argmax(diffs) + 1
    else:
        split_idx = 1

    # Divide into left and right columns
    left_col = [c[1] for c in centers[:split_idx]]
    right_col = [c[1] for c in centers[split_idx:]]

    # Sort each column by y (top -> bottom)
    left_col.sort(key=lambda b: b.bbox[1])
    right_col.sort(key=lambda b: b.bbox[1])

    # For Arabic (RTL), read right column first, then left column
    return right_col + left_col