import sys
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

from keras.models import load_model

from lib import consts as consts
from lib import utils as utils
from lib import data_loader as loader

# To not let numpy print() squish arrays
# np.set_printoptions(threshold=sys.maxsize)

mnist_model = load_model(sys.argv[1])

class App:
    def __init__(self):
        root = tk.Tk()
        self.canvas = tk.Canvas(root, width=consts.WIN_W, height=consts.WIN_H)
        self.canvas.pack()
        self.canvas.old_coords = None
        self.mousedown = False

        self.image = Image.new('L', (consts.WIN_W, consts.WIN_H), (0x0))
        self.imdraw  = ImageDraw.Draw(self.image)

        root.bind('<Motion>', self.draw)
        root.bind('<ButtonPress-1>', self.mouse_down)
        root.bind('<ButtonRelease-1>', self.mouse_up)
        root.bind('<Escape>', self.guess)
        root.bind('<Key>', self.try_save)
        root.bind('<space>', self.clear)
        root.mainloop()

    def draw_circle(self, x, y, r):
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        # self.canvas.create_oval(x0, y0, x1, y1, fill='#000')
        self.canvas.create_oval(x0, y0, x1, y1, fill='#000')
        self.imdraw.rectangle([x0, y0, x1, y1], (255))

    def mouse_up(self, e):
        self.mousedown = False

    def mouse_down(self, e):
        self.mousedown = True

    def try_save(self, e):
        image_path = '{}/{}.png'.format(consts.MY_IMAGES, utils.get_seconds())
        if e.char == '-':
            print('Saving bad guess image to: {}'.format(image_path))
            im = self.image.resize((28, 28), resample=Image.LANCZOS)
            im.save(image_path, 'PNG')

    def guess(self, e):
        im = self.image.resize((28, 28), resample=Image.LANCZOS)
        im.save('/tmp/last.jpg', 'JPEG', quality=99)
        arr = np.array(im)

        if mnist_model.layers[0].name == 'conv2d':
            arr = arr.reshape((1, 28, 28, 1))
        else:
            arr = arr.reshape((1, 28*28))

        predicted = mnist_model.predict_classes(arr)
        if predicted[0] < 10:
            print('My ML guess is: {}'.format(predicted[0]))
        else:
            print('My ML guess is: NaN')

    def clear(self, e):
        self.canvas.delete("all")
        self.imdraw.rectangle((0, 0, consts.WIN_W, consts.WIN_H), fill=(0))

    def draw(self, event):
        if self.mousedown:
            x, y = event.x, event.y
            if self.canvas.old_coords:
                x1, y1 = self.canvas.old_coords
                self.draw_circle(x, y, 13)
            self.canvas.old_coords = x, y

App()
