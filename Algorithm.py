import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import configparser
import os
import time


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
        # self.plt.ion()
        # self.fig, self.ax = plt.subplots()  # Create figure and axes
        # self.fig, self.ax = plt.subplots()  # Create figure and axes
        # self.ax.set_xlabel('x')
        # self.ax.set_ylabel('y')
        # self.ax.set_title('Contour Plot mmwave data')
        plt.pause(0.001)

    def init_contor_plot(self):  # x, and y are 1D arrays and z is a 2D array
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        plt.contourf(self.X, self.Y, self.z, cmap='RdBu_r')
        # self.fig.colorbar(self.ax.contourf(self.X, self.Y, self.z, cmap='RdBu_r'))
        plt.pause(0.001)

    def update_contor_plot(self, z: np.ndarray):
        plt.contourf(self.X, self.Y, z, cmap='RdBu_r')
        plt.pause(0.001)

    def update_contor_plot_with_rect(self, x, y, z, x1, y1, x2, y2):
        self.ax.add_patch(patches.Rectangle((x1, y1), x2, y2,
                          linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()

    def live_contor_plot(self, z: np.ndarray):
        # self.ax.clear()
        self.ax.contourf(self.X, self.Y, z.T, cmap='RdBu_r')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Contour Plot')
        self.fig.colorbar(self.ax.contourf(self.X, self.Y, z.T, cmap='RdBu_r'))
        plt.pause(0.001)

class ConfigFileLoader:
    def __init__(self, config_file_path: str=None) -> None:
        self.config_file_path = config_file_path
        self.config_obj = configparser.ConfigParser()
        self.configParameters = None
        self.dataset_path = None
        self.y_t1 = None
        self.y_t2 = None
        self.x_t1 = None
        self.load_config_file()
        self.extract_parameters()
    
    def load_config_file(self):
        if not os.path.exists(self.config_file_path):
            raise Exception("Config file does not exist")
        self.config_obj.read(self.config_file_path)

    # extract config parameters
    def extract_config_parameters(self):
        self.configParameters = {}
        # for key in self.config_obj['config_parameters']:
        #     self.configParameters[key] = self.config_obj['config_parameters'][key]

        self.configParameters = {'numDopplerBins': int(self.config_obj['config_parameters']['numDopplerBins']),
                                 'numRangeBins': int(self.config_obj['config_parameters']['numRangeBins']),
                                 'rangeResolutionMeters': float(self.config_obj['config_parameters']['rangeResolutionMeters']),
                                 'rangeIdxToMeters': float(self.config_obj['config_parameters']['rangeIdxToMeters']),
                                 'dopplerResolutionMps': float(self.config_obj['config_parameters']['dopplerResolutionMps']),
                                 'maxRange': float(self.config_obj['config_parameters']['maxRange']),
                                 'maxVelocity': float(self.config_obj['config_parameters']['maxVelocity'])}
    
    # extract dataset parameters
    def extract_dataset_parameters(self):
        self.dataset_path = self.config_obj['dataset']['path']

    # extract threshold parameters
    def extract_threshold_parameters(self):
        self.y_t1 = float(self.config_obj['thresholds']['y_t1'])
        self.y_t2 = float(self.config_obj['thresholds']['y_t2'])
        self.x_t1 = float(self.config_obj['thresholds']['x_t1'])

    # extract all parameters
    def extract_parameters(self):
        self.extract_config_parameters()
        self.extract_dataset_parameters()
        self.extract_threshold_parameters()
        print("Config parameters: ", self.configParameters)
        print("Dataset path: ", self.dataset_path)
        print("Thresholds: ", self.y_t1, self.y_t2, self.x_t1)


class Algorithm:
    def __init__(self, config_file_path: str=None, enable_contor_plot: bool = False) -> None:
        self.config_file_path = config_file_path  # path to config file
        self.enable_contor_plot = enable_contor_plot # enable contor plot

        self.config_obj = ConfigFileLoader(config_file_path=self.config_file_path) # config file loader object
        self.configParameters = self.config_obj.configParameters # config parameters
        self.dataset_path = self.config_obj.dataset_path # dataset path

        self.y_t1 = self.config_obj.y_t1 # threshold 1
        self.y_t2 = self.config_obj.y_t2 # threshold 2
        self.x_t1 = self.config_obj.x_t1 # threshold 3

        self.x_axis = None # rangeArray
        self.y_axis = None # dopplerArray
        self.x_mask = None
        self.y_axis = None
        self.data_masked = None
        self.data = None
        self.range_doppler_features = None
        self.contor_plotter = None
        self.num_sections = 8
        self.section_length = None
        self.x_start_index = 0

        self.load_data()
        self.axis_generation()
        self.generate_axis_mask()
        if self.enable_contor_plot:
            self.init_contor_plot()

    # load data from dataset
    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset file does not exist")
        self.range_doppler_features = np.load(
            self.dataset_path, allow_pickle=True)
        self.data = self.range_doppler_features['out_x']

    # generate range and doppler axis
    def axis_generation(self):
        # Generate the range and doppler arrays for the plot
        self.x_axis = np.array(range(
            self.configParameters["numRangeBins"])) * self.configParameters["rangeIdxToMeters"]
        self.y_axis = np.multiply(np.arange(-self.configParameters["numDopplerBins"] / 2, self.configParameters["numDopplerBins"] / 2),
                                   self.configParameters["dopplerResolutionMps"])
    # initialize contor plot
    def init_contor_plot(self):
        # print("Initializing contor plot")
        self.contor_plotter = ContorPlotter(x_axis=self.x_axis, y_axis=self.y_axis)
        self.contor_plotter.init_contor_plot()

    # update contor plot
    def update_contor_plot(self, z):
        # print("Updating contor plot")
        self.contor_plotter.update_contor_plot(z)
    
    def index_threshold(self):
        x_start_index = np.argwhere(self.x_axis > (self.x_t1 * self.x_axis.max()))[0][0]
        y_u_start_index = 0
        y_u_last_index = np.argwhere(self.y_axis > self.y_t1)[0][-1]
        y_l_start_index = np.argwhere(self.y_axis < -self.y_t1)[0][0]
        y_l_start_index = -1
        section_length = (len(x_start_index)-x_start_index) // self.num_sections

    def calculate_max_sum_of_squares(self, matrix):
        # Calculate the maximum sum of squares and the ratio of Max/Mean for each section
        
        # for i in range(self.num_sections):
        #     section = matrix[i * self.section_length: (i + 1) * self.section_length, :]
        #     section_sum_of_squares = np.sum(section ** 2)
        #     result.append(section_sum_of_squares)
        # return result
    
    # def calculate_ratio(self, matrix):
    #     result = []
    #     for i in range(self.num_sections):
    #         section = matrix[i * self.section_length: (i + 1) * self.section_length, :]
    #         section_mean = np.mean(section)
    #         ratio = section_sum_of_squares / section_mean
    #         result.append(ratio)
    #     return result
    # def generate_axis_mask(self):
    #     # Create boolean masks for the specified conditions
    #     self.y_mask = np.logical_or(self.y_axis < -self.y_t1, self.y_axis > self.y_t1)
    #     self.x_mask = self.x_axis > (self.x_t1 * self.x_axis.max())
    
    # def apply_mask(self, matrix):
    #     # Apply the masks to the matrix
    #     self.data_masked = np.copy(matrix)
    #     self.data_masked[self.y_mask, :] = 0
    #     self.data_masked[:, self.x_mask] = 0
    
    # def get_section_length(self):
    #     # Divide the unmasked segment into equal sections
    #     self.x_start_index = len(self.x_axis)-sum(self.x_mask)
    #     self.section_length = sum(self.x_mask) // self.num_sections

    # def calculate_max_sum_of_squares(self, matrix):
    #     # Calculate the maximum sum of squares and the ratio of Max/Mean for each section
    #     result = []
    #     for i in range(self.num_sections):
    #         section = matrix[i * self.section_length: (i + 1) * self.section_length, :]
    #         section_sum_of_squares = np.sum(section ** 2)
    #         result.append(section_sum_of_squares)
    #     return result

    

    ###
    # Apply the masks to the matrix
        # y_mask = np.logical_or(dopplerArray < -0.2, dopplerArray > 0.2)
        # x_mask = rangeArray > (0.2 * rangeArray.max())
        # masked_matrix = np.copy(matrix)
        # masked_matrix[y_mask, :] = 0
        # masked_matrix[:, x_mask] = 0

        # # Divide the unmasked segment into equal sections
        # num_sections = 4
        # section_length = masked_matrix.shape[0] // num_sections

        # # Calculate the maximum sum of squares and the ratio of Max/Mean for each section
        # max_sum_of_squares = []
        # ratios = []

        # for i in range(num_sections):
        #     section = masked_matrix[i * section_length: (i + 1) * section_length, :]
        #     section_sum_of_squares = np.sum(section ** 2)
        #     max_sum_of_squares.append(section_sum_of_squares)
        #     section_mean = np.mean(section)
        #     ratio = section_sum_of_squares / section_mean
        #     ratios.append(ratio)

        # # Compare the ratios with a predefined threshold
        # threshold = 10

        # # Check if any of the ratios exceed the threshold
        # exceed_threshold = any(ratio > threshold for ratio in ratios)

        # # Print the ratios and the result of the comparison
        # print("Ratios:", ratios)
        # print("Exceed threshold:", exceed_threshold)

    def run(self):
        for i in self.data:
            print(i.shape)
            if self.enable_contor_plot:
                self.update_contor_plot(i)
            time.sleep(0.001)

if __name__ == "__main__":
    algorithm = Algorithm(config_file_path="config.ini", enable_contor_plot=True)
    algorithm.init_contor_plot()
    algorithm.run()
    # feature_x = np.arange(0, 50, 2)
    # feature_y = np.arange(0, 50, 3)
    # contor_plotter = ContorPlotter(x_axis=feature_x, y_axis=feature_y)
    # contor_plotter.init_contor_plot()
    
