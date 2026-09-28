def calculate_pothole_severity(box_xyxy, frame_shape=None):
    """
    Calculates pothole severity level (1=Low, 2=Medium, 3=High)
    based on relative bounding box area (PDF Spec Section 5A).
    """
    width = box_xyxy[2] - box_xyxy[0]
    height = box_xyxy[3] - box_xyxy[1]
    area = width * height

    if area > 45000:
        return 3  # High severity
    elif area > 18000:
        return 2  # Medium severity
    else:
        return 1  # Low severity
