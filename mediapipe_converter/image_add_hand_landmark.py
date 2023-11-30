import cv2
import mediapipe as mp

# 手勢辨識模型
mpHand = mp.solutions.hands

# 取得所有手
hands = mpHand.Hands()

# mdeiaPipe 繪圖工具
mpDraw = mp.solutions.drawing_utils

# 注意 色彩格式為 BGR
nodeStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=5,  circle_radius=1)
lineStyle = mpDraw.DrawingSpec(color=(0,255,0), thickness=10)

img = cv2.imread("D:/Users/Naukri/Desktop/hand.jpg")

# 將BGR格式轉換為RGB格式
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = hands.process(rgb_image)

hadns = result.multi_hand_landmarks 

# print(hadns)

if hadns:
    for hand in hadns:
        # 繪製含有關節點特效的圖片
        mpDraw.draw_landmarks(img, hand, mpHand.HAND_CONNECTIONS, nodeStyle, lineStyle)
        # 取得所有關節點的資訊 (i = 索引 0-20, lm.x = x軸 ratio, lm.y = y軸 ratio)
        for i, lm in enumerate(hand.landmark):
            print(i, lm.x, lm.y)
    # 輸出成圖片        
    cv2.imwrite("D:/Users/Naukri/Desktop/handD.jpg", img)
