from PIL import Image
from net.frcnn import FRCNN

frcnn = FRCNN()

img = '/home/zhaohoj/Pictures/getImage.png'
try:
    image = Image.open(img)
    # -------------------------------------#
    #   转换成RGB图片，可以用于灰度图预测。
    # -------------------------------------#
    image = image.convert("RGB")
except:
    print('Open Error! Try again!')
else:
    r_image = frcnn.detect_image(image)
    r_image.show()
