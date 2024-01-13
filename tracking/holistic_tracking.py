import cv2
import mediapipe as mp
import time

# 导入模块
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 初始模型
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True,  # 使用attention mesh, 对眼睛和嘴唇建模更精细
)

def process_frame(img):
    start_time = time.time()  # 记录开始时间

    h, w, _ = img.shape  # 获取图像宽高
    img = cv2.flip(img, 1)  # 镜像翻转图像，使图中左右手与真实左右手对应
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

    results = holistic.process(img_RGB)
    
    mp_drawing.draw_landmarks(
        img,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    # mp_drawing.draw_landmarks(
    #     img,
    #     results.face_landmarks,
    #     mp_holistic.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        img,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        img,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    # TODO: 使用类似于hand_tracking中的方法来标记眼睛等部位
    end_time = time.time()  # 记录结束时间
    FPS = 1/(end_time - start_time)  # 计算每秒处理图像帧数

    # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    img = cv2.putText(img, 'FPS: '+str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
    return img

def main():
    cap = cv2.VideoCapture(0)
    cap.open(0)

    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        if not success:
            break
        
        ## 处理帧函数
        frame = process_frame(frame)
        
        # 展示处理后的三通道图像
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
            break
        
    cap.release()  # 关闭摄像头
    cv2.destroyAllWindows()  # 关闭图像窗口

if __name__ == "__main__":
    main()
