# Program to detect faces in an image using face_recognition algorithm

from PIL import Image
from PIL import ImageDraw
import face_recognition

image = face_recognition.load_image_file('2.jpg')
face_locations = face_recognition.face_locations(image)
print('number of faces: ', len(face_locations))

pil_Image = Image.fromarray(image)
for (top, right, bottom, left) in face_locations:
    draw = ImageDraw.Draw(pil_Image)
    draw.rectangle([left, top, right, bottom], outline = 'red', width = 2)


pil_Image.show()