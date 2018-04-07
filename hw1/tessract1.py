from PIL import Image
import pytesseract

im = Image.open("images/51H1M.png")

text = pytesseract.image_to_string(im, lang = 'eng')

print(text)