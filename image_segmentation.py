from pathlib import Path
import cv2
import numpy as np
from fastai.learner import load_learner
import detectorModel


def preprocess_image(image_path):
    """Setup image for processing"""
    # Read and threshold image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Use morphological operations to connect broken parts
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated


def should_merge(box1, box2):
    """Determine if two bounding boxes should be merged"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return x1 <= x2 <= x1+w1 or x2 < x1 < x2+w2


def merge_boxes(boxes):
    """Merge multiple bounding boxes into one"""
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[0] + box[2] for box in boxes)
    y_max = max(box[1] + box[3] for box in boxes)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def segment_symbols(original_img, thresh_img):
    """Parse equation into set of symbols"""
    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes for all contours and sort them by x-coordinate (left-to-right)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes.sort(key=lambda b: b[0])  # Sort by x-coordinate

    # Group nearby boxes
    groups = []
    used_indices = set()

    for i, box in enumerate(boxes):
        if i in used_indices:
            continue

        current_group = [box]
        used_indices.add(i)

        # Check nearby boxes
        for j, other_box in enumerate(boxes[i + 1:], start=i + 1):
            if j in used_indices:
                continue

            if should_merge(box, other_box):
                current_group.append(other_box)
                used_indices.add(j)

        groups.append(current_group)

    # Create merged symbols from ORIGINAL image
    symbols = []
    for group in groups:
        if len(group) == 1:
            x, y, w, h = group[0]
        else:
            x, y, w, h = merge_boxes(group)

        # Extract with padding FROM ORIGINAL IMAGE
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(original_img.shape[1], x + w + padding)
        y_end = min(original_img.shape[0], y + h + padding)

        symbol = original_img[y_start:y_end, x_start:x_end]

        symbols.append(symbol)

    return symbols

def symbol_conversion(symbols):
    """Convert symbol to operator or digit using ML model"""
    equation = ""
    model = load_learner('model/model.pkl')
    for i, symbol in enumerate(symbols):
        file_path = Path("symbols") / f"symbol_{i+1}.png"
        cv2.imwrite(str(file_path), symbol)  # Save image
        thresh = preprocess_image(file_path)
        thresh = cv2.bitwise_not(thresh)
        cv2.imwrite(str(file_path), thresh)
        pred_class, pred_idx, probs = model.predict(file_path)
        symbol = detectorModel.detector(file_path)
        equation += str(symbol)
    return equation

def find_equation(image_path):
    """Convert original screenshot to equation"""
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read original
    thresh = preprocess_image(image_path)  # Get processed version
    symbols = segment_symbols(original_img, thresh)
    return symbol_conversion(symbols)
