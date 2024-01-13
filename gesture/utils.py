def calculate_distance(result, landmark1, landmark2):
    """
    Calculate the relative distance between two landmarks
    """
    x1, y1 = result.landmark[landmark1].x, result.landmark[landmark1].y
    x2, y2 = result.landmark[landmark2].x, result.landmark[landmark2].y
    
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance

def calculate_center(result):
    """
    Calculate the coordinates of the center point of the target.
    """
    x_vals = [landmark.x for landmark in result.landmark]
    y_vals = [landmark.y for landmark in result.landmark]
    center_x = sum(x_vals) / len(result.landmark)
    center_y = sum(y_vals) / len(result.landmark)
    return center_x, center_y

def is_finger_folded(hand, finger_base_index, finger_tip_index, folded_threshold):
    """
    Check if a finger is folded.
    """
    # 计算手指基部和尖端之间的距离
    distance = calculate_distance(hand, finger_base_index, finger_tip_index)

    return distance < folded_threshold

def check_ok_gesture(hand, threshold=0.1, folded_threshold=0.13):
    """
    Check the OK gesture while calculating the distance between thumb tip and index finger tip.
    """
    thumb_tip_index = 4
    index_finger_tip_index = 8
    middle_finger_tip_index = 12
    ring_finger_tip_index = 16
    little_finger_tip_index = 20
    
    distance = calculate_distance(hand, thumb_tip_index, index_finger_tip_index)
    
    middle_folded = is_finger_folded(hand, 9, middle_finger_tip_index, folded_threshold)
    ring_folded = is_finger_folded(hand, 13, ring_finger_tip_index, folded_threshold)
    little_folded = is_finger_folded(hand, 17, little_finger_tip_index, folded_threshold)

    return distance < threshold and not middle_folded and not ring_folded and not little_folded

def check_thumb_up(hand):
    """
    Check if the thumb up gesture is being made.
    """
    thumb_tip_index = 4
    thumb_mcp_index = 2  # MCP（掌指关节）的索引
    thumb_tip = hand.landmark[thumb_tip_index]
    thumb_mcp = hand.landmark[thumb_mcp_index]

    # 检查大拇指是否伸直
    # 这可以通过比较大拇指尖和MCP关节的Y坐标来判断
    thumb_straight = thumb_tip.y < thumb_mcp.y

    # 检查大拇指尖是否高于其他手指的PIP关节
    # 我们可以检查每个手指的PIP关节
    fingers_pip_indices = [6, 10, 14, 18]  # 分别是食指、中指、无名指和小指的PIP关节索引
    thumb_above_fingers = all(thumb_tip.y < hand.landmark[pip_index].y for pip_index in fingers_pip_indices)

    return thumb_straight and thumb_above_fingers
