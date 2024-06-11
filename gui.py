import tkinter as tk
import numpy as np
from PIL import Image, ImageGrab
import cv2
import torch, mnist
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class PaintApp:
    def __init__(self, root):

        self.root = root
        self.color = "white"
        self.drawing_area = tk.Canvas(self.root, width=784, height=784, bg="black")
        self.drawing_area.pack()
        self.last_x = 0
        self.last_y = 0
        self.drawing_area.bind("<B1-Motion>", self.draw)
        self.drawing_area.bind("<Button-1>", self.setDraw)
        self.drawing_area.bind("<Button-3>", self.setErase)
        self.drawing_area.bind("<B3-Motion>", self.clear)
        self.drawing_area.bind("<Button-2>", self.clearAll)
        self.networkF = torch.load("mnistConv01.pth")
        self.network = mnist.Network()
        self.network.load_state_dict(self.networkF)
        self.label = tk.Label(self.root, text="Prediction: ")
        self.label.pack()

    def setDraw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def setErase(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        self.drawing_area.create_oval(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            width=20,
            fill=self.color,
            outline=self.color,
        )
        self.last_x = event.x
        self.last_y = event.y

    def clear(self, event):
        self.drawing_area.create_oval(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            width=20,
            fill="black",
            outline="black",
        )
        self.last_x = event.x
        self.last_y = event.y

    def clearAll(self, event):
        self.drawing_area.delete("all")


    def get_np_array(self):
        x = self.root.winfo_rootx() + self.drawing_area.winfo_x()
        y = self.root.winfo_rooty() + self.drawing_area.winfo_y()
        x1 = x + self.drawing_area.winfo_width()
        y1 = y + self.drawing_area.winfo_height()
        img = ImageGrab.grab((x, y, x1, y1))
        img = img.resize((28, 28))
        img = img.convert("L")
        img = np.array(img)
        return img

    def update(self):
        image = self.get_np_array()
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        image = transform(image)
        image = image.unsqueeze(0)
        output = self.network(image)

        output = torch.nn.functional.softmax(output, dim=1)

        self.label.config(text=f"0: {output[0][0].item() * 100:0.2f}% 1: {output[0][1].item() * 100:0.2f}% 2: {output[0][2].item() * 100:0.2f}% 3: {output[0][3].item() * 100:0.2f}% 4: {output[0][4].item() * 100:0.2f}% 5: {output[0][5].item() * 100:0.2f}% 6: {output[0][6].item() * 100:0.2f}% 7: {output[0][7].item() * 100:0.2f}% 8: {output[0][8].item() * 100:0.2f}% 9: {output[0][9].item() * 100:0.2f}")


root = tk.Tk()

app = PaintApp(root)

while True:
    app.update()
    root.update()
