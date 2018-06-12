from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import tkinter as tk
from preprocessors import simple_cnn_preprocessor
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

DRAWING_AREA = (280, 280)
predictions = [0 for x in range(10)]
ax = None

# load the model
with open('./model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights('./model.h5')


def new_image():
    return Image.new("L", DRAWING_AREA)


image = new_image()
drawer = ImageDraw.Draw(image)


# helper functions
def _paint_(event):
    global predictions, ax
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="#FFFFFF", outline="")
    drawer.ellipse([x1, y1, x2, y2], fill=255, outline=255)
    temp_image = image.copy()
    temp_image.thumbnail((28, 28), Image.ANTIALIAS)
    temp_image = img_to_array(temp_image)
    temp_image = simple_cnn_preprocessor(temp_image)

    predictions = model.predict(temp_image)[0].tolist()
    print(predictions)


def _update_(event):
    plt.cla()
    plt.bar(range(0, 10), predictions)
    plt.draw()


def _clear_():
    global image, drawer, predictions
    canvas.delete("all")
    predictions = [0 for x in range(10)]
    image = new_image()
    drawer = ImageDraw.Draw(image)
    plt.cla()
    plt.draw()


# create the canvas
master = tk.Tk()
master.title("Digit recognizer")
canvas = tk.Canvas(master,
                   width=DRAWING_AREA[0],
                   height=DRAWING_AREA[1])
canvas.configure(background="black")
canvas.pack()
canvas.bind("<B1-Motion>", _paint_)
canvas.bind("<ButtonRelease-1>", _update_)

clear = tk.Button(master, text="CLEAR", command=_clear_)
clear.pack(side=tk.BOTTOM)

plt.xlabel('Digits')
plt.ylabel('Probability')
plt.title('Predictions')
plt.bar(range(0, 10), predictions)
plt.show()

tk.mainloop()
