import numpy as np
import cv2


def convert(video_path):
    cap = cv2.VideoCapture(video_path)

    success, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    while cap.isOpened():
        success, frame2 = cap.read()

        if not success:
            break

        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, flow=None, 
                pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next

    cv2.destroyAllWindows()
    cap.release()

convert('../~data/test/1-converted.mp4')