from PIL import Image
import glob

pgm_image_list = glob.glob('*.pgm')

for filename in pgm_image_list:
    img = Image.open(filename)
    img.save(filename[:-4] + '.jpg')
