import numpy as np
import os
import time
from utils.config_reader import ConfigFileLoader
from utils.contour_plotter import ContorPlotter
from utils.utils import increment_path
from utils.csv_handler import CSVHandler
from pathlib import Path
File = Path(__file__).resolve()
ROOT = File.parents[1] 

class Algorithm:
    def __init__(self, config_file_path: str=None) -> None:
        self.config_file_path = config_file_path  # path to config file

        self.config_obj = ConfigFileLoader(config_file_path=self.config_file_path) # config file loader object
        self.configParameters = self.config_obj.configParameters # config parameters
        self.dataset_path = self.config_obj.dataset_path # dataset path

        self.y_t1 = self.config_obj.y_t1 # threshold 1
        self.y_t2 = self.config_obj.y_t2 # threshold 2
        self.x_t1 = self.config_obj.x_t1 # threshold 3
        self.threshold = self.config_obj.threshold # threshold
        self.plot_parameters = self.config_obj.plot_parameters # plot parameters
        self.csv_parameters = self.config_obj.csv_parameters # csv parameters
        self.Project = self.config_obj.save_dir # save parameters
        self.save_dir = increment_path(Path(self.Project) / 'runs', exist_ok=False, mkdir=True)  # increment run
        print("new path: ", self.save_dir)

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
        self.y_t_start_index = 0
        self.y_t_last_index = 0
        self.y_b_start_index = 0
        self.y_b_start_index = 0
        self.section_length = 8
        self.result = {}
        self.decision = None
        self.load_data()
        self.axis_generation()
        self.index_threshold()
        if self.plot_parameters['enable_plot']:
            self.init_contor_plot()
        if self.csv_parameters['enable_csv']:
            self.init_csv_writer()
            # self.add_rect()

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
        print(self.x_axis)
        print(self.y_axis)
    # initialize contor plot
    def init_contor_plot(self):
        self.contor_plotter = ContorPlotter(x_axis=self.x_axis, y_axis=self.y_axis)
        self.contor_plotter.init_contor_plot()

    # update contor plot
    def update_contor_plot(self, z):
        # print("Updating contor plot")
        self.contor_plotter.update_contor_plot(z)
    
    def init_csv_writer(self):
        self.csv_writer = CSVHandler(path=self.save_dir, csv_file_name=self.csv_parameters['csv_filename'])
        self.csv_writer.init_csv_writer()
    
    def index_threshold(self):
        self.x_start_index = np.argwhere(self.x_axis > (self.x_t1 * self.x_axis.max()))[0][0]
        self.y_t_start_index = 0
        self.y_t_last_index = len(self.y_axis)-np.short(np.argwhere(self.y_axis > self.y_t1))[0,-1]
        self.y_b_start_index = len(self.y_axis)-np.short(np.argwhere(self.y_axis < self.y_t2))[-1,-1]
        self.y_b_last_index = len(self.y_axis)-1
        self.section_length = (len(self.x_axis)-self.x_start_index) // self.num_sections

    def add_rect(self):
        for i in range(self.num_sections):
            x1 = self.x_axis[self.x_start_index + i * self.section_length]
            x2 = self.x_axis[self.x_start_index + (i + 1) * self.section_length]
            y1 = self.y_axis[self.y_t_start_index]
            y2 = self.y_axis[self.y_t_last_index]
            self.contor_plotter.add_rect(y1, x1, (y2-y1), (x2-x1))

            x1 = self.x_axis[self.x_start_index + i * self.section_length]
            x2 = self.x_axis[self.x_start_index + (i + 1) * self.section_length]
            y1 = self.y_axis[self.y_b_start_index]
            y2 = self.y_axis[self.y_b_last_index]
            self.contor_plotter.add_rect(y1, x1, (y2-y1), (x2-x1))
            # self.contor_plotter.add_rect(0.2, 15, 0.8, 20)

    def operation(self, matrix):
        self.result['t'] = np.zeros(self.num_sections)
        self.result['b'] = np.zeros(self.num_sections)

        for i in range(self.num_sections):
            temp_t = matrix[self.y_t_start_index:self.y_t_last_index, self.x_start_index + i * self.section_length: self.x_start_index + (i + 1) * self.section_length]
            temp_b= matrix[self.y_b_start_index:self.y_b_last_index, self.x_start_index + i * self.section_length: self.x_start_index + (i + 1) * self.section_length]
            sum_of_squares_t = np.sum(temp_t ** 2)
            mean_t = np.mean(temp_t)
            ratio_t = sum_of_squares_t / mean_t
            self.result['t'][i] = ratio_t
            sum_of_squares_b = np.sum(temp_b ** 2)
            mean_m = np.mean(temp_b)
            ratio_m = sum_of_squares_b/ mean_m
            self.result['b'][i] = ratio_m
    
    def max_ratio(self):
        return max(max(self.result['t']), max(self.result['b']))
    
    def run(self):
        for file_name, i in enumerate(self.data):
            try:
                self.operation(i)
                max_ratio = self.max_ratio()
                self.decision = "Human Detected" if (max_ratio > self.threshold) else "No Human Detected"
                if self.plot_parameters['enable_plot']:
                    self.update_contor_plot(i)
                    if self.plot_parameters['add_text_rect']:
                        self.add_rect()
                        self.contor_plotter.update_text("Decision: " + str(self.decision) + "\nRatio: " + str(round(max_ratio,2))+"\nThreshold: "+str(self.threshold))
                    if self.plot_parameters['save_plot']:
                        self.contor_plotter.save_plot(self.save_dir, file_name)
                if self.csv_parameters['enable_csv']:
                    self.csv_writer.write_csv_row([max_ratio]) # write to csv file 
                print("Decision: ", self.decision)
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("Terminated by user.... Bye")
                break

if __name__ == "__main__":
    algorithm = Algorithm(config_file_path="configfile/config_outdoor.ini")
    algorithm.run()
    
