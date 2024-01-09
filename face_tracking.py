import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh

# 初始化面部模型
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True,  # 使用Attention mesh
    )

# 导入绘图函数
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_frame(img):
    start_time = time.time()  # 记录开始时间

    h, w = img.shape[0], img.shape[1]  # 获取图像宽高
    img = cv2.flip(img, 1)  # 镜像翻转图像，使图中左右手与真实左右手对应
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_results = face_mesh.process(img_RGB)
    required_landmarks = []
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
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
        scaler = 1
        end_time = time.time()  # 记录结束时间
        FPS = 1/(end_time - start_time)  # 计算每秒处理图像帧数

        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    return img, required_landmarks


def main():
    cap = cv2.VideoCapture(0)
    cap.open(0)

    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        if not success:
            break

        # 处理帧函数
        frame, required_landmarks = process_frame(frame)

        # 展示处理后的三通道图像
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
            break

    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭图像窗口


if __name__ == "__main__":
    main()
