import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class ContorPlotter:
    def __init__(self, x_axis, y_axis) -> None:
        self.x = x_axis  # 1D array
        self.y = y_axis  # 1D array
        self.z = np.zeros((len(y_axis), len(x_axis)))  # 2D array
        # self.z = np.random.rand(len(y_axis), len(x_axis))  # 2D array
        print("shape of x: ", self.x.shape)
        print("shape of y: ", self.y.shape)
        print("shape of z: ", self.z.shape)
        self.X = None  # 2D array
        self.Y = None  # 2D array
        # plt.contourf(self.x, self.y, self.z, cmap='RdBu_r')
        plt.xlabel('Doopler Range')
        plt.ylabel('Doopler Velocity')
        plt.pause(0.001)
        self.text = plt.text(self.x[0], self.y[0], "Decision: ", fontsize=12)

    def init_contor_plot(self):  # x, and y are 1D arrays and z is a 2D array
        self.X, self.Y = np.meshgrid(self.x, self.y)
        plt.contourf(self.X, self.Y, self.z, cmap='RdBu_r')
        # self.fig.colorbar(self.ax.contourf(self.X, self.Y, self.z, cmap='RdBu_r'))
        # plt.pause(0.001)

    def update_contor_plot(self, z: np.ndarray):
        plt.clf()
        plt.contourf(self.X, self.Y, z, cmap='RdBu_r')
        plt.pause(0.001)

    def add_rect(self, x1, y1, x2, y2):
        # Create a Rectangle patch
        rect = patches.Rectangle((y1, x1),y2 , x2, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        # plt.pause(0.001)
    
    def add_text(self, x, y, text=[]):
        # self.text = plt.text(self.x[0], self.y[0], "Decision: ", fontsize=12)
        plt.text(x, y, text, fontsize=12)
        # plt.pause(0.001)
    
    def update_text(self,text):
        plt.text(self.x[1], self.y[1], text, fontsize=12, color='#800080')
        # self.text.set_text(text)
        # plt.text(text, fontsize=12)
        plt.pause(0.001)
    
    def save_plot(self, path, file_name):
        plt.savefig(os.path.join(path, str(file_name)+'.png'))
    