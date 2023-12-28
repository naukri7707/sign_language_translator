
import cv2

def random_flip(img, seed: int = 0):
    if not seed % 2 == 0:
        img = cv2.flip(img, 1)
    
    return img

def random_rotation(img, range: (int, int) = (-5, 5), seed: int = 0):
    min, max = range

    size = max - min + 1

    idx = seed % size

    angle = min + idx

    h, w, _ = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))   
    
    return img

def random_zoom(img, range: (int, int) = (90, 110), seed: int = 0):
    min, max = range

    size = max - min + 1

    idx = seed % size

    zoom_factor = (min + idx) / 100

    # 取得圖片尺寸
    height, width = img.shape[:2]

    # 新的寬度和高度
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # 執行縮放
    zoomed_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return zoomed_image