import cv2
import sys
import json

if (__name__ == "__main__"):
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    convert_to_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output.png', image)

    data = {
        "test1": "1",
        "test2": "2"
    }
    print(json.dumps(data))