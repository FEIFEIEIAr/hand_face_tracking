import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)   # 0 for webcam, 1 for external camera

# 初始化模型
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,  # Video or image
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,  # 使用Attention mesh
)

# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_frame(img):
    # 记录该帧开始处理的时间
    start_time = time.time()

    h, w, _ = img.shape  # 获取图像宽高
    img = cv2.flip(img, 1)  # 镜像翻转图像，使图中左右手与真实左右手对应
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

    # 处理手部信息
    results = hands.process(img_RGB)
    if results.multi_hand_landmarks:  # 如果有检测到手
        hand_info = ''
        index_finger_tip_info = ''
        # 遍历检测到的所有手
        for hand_idx in range(len(results.multi_hand_landmarks)):
            # 获取手的21个landmark坐标
            hand = results.multi_hand_landmarks[hand_idx]

            # 可视化landmark连线
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            # 记录手的信息
            hand_label = results.multi_handedness[hand_idx].classification[0].label
            hand_info = f"{hand_idx} {hand_label}\n"
            
            # 获取手腕根部深度坐标
            z0 = hand.landmark[0].z

            for i in range(21):  # 遍历该手的21个关键点
                # 获取3D坐标
                x = int(hand.landmark[i].x * w)
                y = int(hand.landmark[i].y * h)
                z = hand.landmark[i].z
                depth = z0 - z

                radius = max(int(5 * (1 + depth*10)), 0)  # 用圆的大小反映深度

                if i == 0:  # 手腕
                    img = cv2.circle(img, (x, y), radius, (0, 0, 255), -1)
                if i == 8:  # 食指指尖
                    img = cv2.circle(img, (x, y), radius, (41, 27, 166), -1)
                    # 食指指尖相对于手腕的深度
                    index_finger_tip_info += f'{hand_idx}-{depth:.2f}'
                if i in [4, 12, 16, 20]:  # 指尖（除食指指尖）
                    img = cv2.circle(img, (x, y), radius, (223, 213, 176), -1)
                if i in [2, 6, 10, 14, 18]:  # 第一指节
                    img = cv2.circle(img, (x, y), radius, (140, 193, 102), -1)
                if i in [3, 7, 11, 15, 19]:  # 第二指节
                    img = cv2.circle(img, (x, y), radius, (195, 236, 249), -1)
                if i in [1, 5, 9, 13, 17]:  # 指根
                    img = cv2.circle(img, (x, y), radius, (211, 194, 209), -1)
        
        # 将信息显示在图像上
        img = cv2.putText(img, hand_info, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        img = cv2.putText(img, index_finger_tip_info, (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
                    
    # 处理面部信息
    face_results = face_mesh.process(img_RGB)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # mp_drawing.draw_landmarks(
            #     image=img,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

        end_time = time.time()  # 记录结束时间
        FPS = 1/(end_time - start_time)  # 计算每秒处理图像帧数

        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, 'FPS: '+str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
    return img


def run():
    cap = cv2.VideoCapture(0)
    cap.open(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame = process_frame(frame)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break

    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭图像窗口

if __name__ == '__main__':
    run()
