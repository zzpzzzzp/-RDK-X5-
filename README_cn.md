import cv2
import dlib
import numpy as np
import math
import serial
import time

# 初始化串口通信
def init_serial(port, baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # 等待串口初始化
        return ser
    except serial.SerialException as e:
        print(f"串口初始化失败: {e}")
        return None

# 计算两点之间的角度
def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# 主函数
def main():
    # 初始化串口
    ser = init_serial('COM3')  # 根据实际情况修改串口号
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 初始化dlib的人脸检测器和面部关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # 初始化手部检测器（使用OpenCV的Haar级联）
    hand_cascade = cv2.CascadeClassifier('hand.xml')  # 需要下载或训练手部检测器
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = detector(gray)
        
        for face in faces:
            # 获取面部关键点
            landmarks = predictor(gray, face)
            
            # 提取肩部和肘部关键点（简化版，实际应用可能需要更复杂的模型）
            # 这里使用面部关键点近似表示肩部
            left_shoulder = (landmarks.part(0).x, landmarks.part(27).y + 50)
            right_shoulder = (landmarks.part(16).x, landmarks.part(27).y + 50)
            
            # 假设肘部位置（需要更精确的模型）
            left_elbow = (left_shoulder[0] - 50, left_shoulder[1] + 100)
            right_elbow = (right_shoulder[0] + 50, right_shoulder[1] + 100)
            
            # 计算肩部到肘部的角度
            left_shoulder_angle = calculate_angle(left_shoulder, left_elbow)
            right_shoulder_angle = calculate_angle(right_shoulder, right_elbow)
            
            # 绘制关键点和连线
            cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_elbow, 5, (0, 0, 255), -1)
            cv2.circle(frame, right_elbow, 5, (0, 0, 255), -1)
            cv2.line(frame, left_shoulder, left_elbow, (255, 0, 0), 2)
            cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 2)
            
            # 在图像上显示角度
            cv2.putText(frame, f"Left Shoulder: {left_shoulder_angle:.1f} degrees", 
                       (left_shoulder[0], left_shoulder[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right Shoulder: {right_shoulder_angle:.1f} degrees", 
                       (right_shoulder[0], right_shoulder[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 通过串口发送角度数据
            if ser:
                try:
                    data = f"LSA:{left_shoulder_angle:.1f},RSA:{right_shoulder_angle:.1f}\n"
                    ser.write(data.encode())
                except serial.SerialException as e:
                    print(f"串口通信错误: {e}")
        
        # 检测手部
        hands = hand_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            hand_center = (x + w//2, y + h//2)
            
            # 计算手部角度（简化版）
            hand_angle = calculate_angle((x, y), (x+w, y))
            cv2.putText(frame, f"Hand Angle: {hand_angle:.1f} degrees", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 通过串口发送手部角度
            if ser:
                try:
                    data = f"HA:{hand_angle:.1f}\n"
                    ser.write(data.encode())
                except serial.SerialException as e:
                    print(f"串口通信错误: {e}")
        
        # 显示结果
        cv2.imshow('Arm and Hand Tracking', frame)
        
        # 按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    if ser:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    
