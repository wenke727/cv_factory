import os
import numpy as np


""" Byte track """
def correct_tlwh(tlwh, image_width=np.float('inf'), image_height=np.float('inf')):
    """
    Corrects the tlwh bounding box format if top-left coordinates are negative.

    Parameters:
        tlwh (tuple): A tuple (top-left x, top-left y, width, height) of the bounding box.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        tuple: A corrected bounding box (top-left x, top-left y, width, height).
    """
    x, y, w, h = tlwh

    # Ensure the top-left corner is within the image boundaries
    new_x = int(max(0, x))
    new_y = int(max(0, y))

    # Adjust width and height if the original x or y were negative
    if x < 0:
        w += x  # Reduce width by the amount x was out of bounds
    if y < 0:
        h += y  # Reduce height by the amount y was out of bounds

    # Ensure width and height do not exceed image boundaries
    w = int(min(w, image_width - new_x))
    h = int(min(h, image_height - new_y))

    return (new_x, new_y, w, h)

def parse_mot_results(filename, skip_frame_num=5):
    basename = os.path.basename(filename)
    
    results = []
    with open(filename, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split each line by comma or space (depends on your file format)
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue  # Ensure there are enough columns in this line

            # Convert each part to the appropriate type
            frame_id = int(parts[0])
            if frame_id % skip_frame_num != 0:
                continue
            
            track_id = int(parts[1])
            bbox_x = float(parts[2])
            bbox_y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            confidence = float(parts[6]) if len(parts) > 6 else None
            class_id = int(parts[7]) if len(parts) > 7 else None
            visibility = float(parts[8]) if len(parts) > 8 else None

            # Create a dictionary for the current tracking result
            track_result = {
                "camera": basename,
                'frame_id': frame_id,
                'track_id': track_id,
                'tlwh': correct_tlwh((bbox_x, bbox_y, width, height)),
                'confidence': confidence,
            }
            if class_id is not None:
                track_id[class_id] = class_id
            if visibility is not None:
                track_id[visibility] = visibility

            # Add the dictionary to the results list
            results.append(track_result)

    return results

