import cv2

def scan_cameras(max_tested=4):
    available_indices = []
    for i in range(max_tested):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"Index {i}: Active ({w}x{h})")
                available_indices.append(i)
            temp_cap.release()
        else:
            print(f"Index {i}: Not available")
    return available_indices

scan_cameras()