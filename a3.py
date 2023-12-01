import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_threshold(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask

def smoothing(mask):
    smoothed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return smoothed_mask

def superimpose_images(background, foreground, mask):
    inverted_mask = cv2.bitwise_not(mask)

    subject = cv2.bitwise_and(foreground, foreground, mask=inverted_mask)

    base = cv2.bitwise_and(background, background, mask=mask)

    result = cv2.add(base, subject)

    return result

def drawShape(image):
    shape = np.zeros_like(image)
    vertices = np.array([[100, 100], [200, 50], [300, 100], [250, 200]], np.int32)
    vertices = vertices.reshape((-1, 1, 2))

    cv2.polylines(shape, [vertices], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.fillPoly(shape, [vertices], color=(66, 135, 245))

    return shape

def main():
    photo1 = cv2.imread('bottle.jpg')
    photo2 = cv2.imread('skull.jpg')

    mask = intensity_threshold(photo1, 150)

    smoothed_mask = smoothing(mask)

    result = superimpose_images(photo2, photo1, smoothed_mask)

    shape = drawShape(result)

    finalImage = cv2.add(result, shape)

    cv2.imwrite('bottleONskull.jpg', finalImage)
    plt.imshow(cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()
