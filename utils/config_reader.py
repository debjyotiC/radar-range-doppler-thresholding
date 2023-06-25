import configparser
import os


class ConfigFileLoader:
    def __init__(self, config_file_path: str=None) -> None:
        self.config_file_path = config_file_path
        self.config_obj = configparser.ConfigParser()
        self.configParameters = None
        self.dataset_path = None
        self.y_t1 = None # top y axis vale
        self.y_t2 = None # bottom y axis value
        self.x_t1 = None # x axis value
        self.threshold = None # threshold
        self.load_config_file()
        self.extract_parameters()
        self.result = {}
    
    def load_config_file(self):
        if not os.path.exists(self.config_file_path):
            raise Exception("Config file does not exist")
        self.config_obj.read(self.config_file_path)

    # extract config parameters
    def extract_config_parameters(self):
        self.configParameters = {}
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
        self.threshold = float(self.config_obj['thresholds']['threshold'])
    
    def extract_plot_parameters(self):
        self.plot_parameters = {}
        for key in self.config_obj['plot_parameters']:
            self.plot_parameters[key] = eval(self.config_obj['plot_parameters'][key])
    
    # def extract_save_parameters(self):
    #     self.save_parameters = {}
    #     self.save_parameters["save_plot"] = eval(self.config_obj['save_parameters']["save_plot"])
    #     self.save_parameters["save_dir"] = self.config_obj['save_parameters']["save_dir"]
    
    def extract_csv_parameters(self):
        self.csv_parameters = {}
        self.csv_parameters["enable_csv"] = eval(self.config_obj['csv_parameters']["enable_csv"])
        self.csv_parameters["csv_filename"] = self.config_obj['csv_parameters']["csv_filename"]

    def extract_extra_parameters(self):
        self.save_dir = self.config_obj['save_dir']["save_dir"]

    # extract all parameters
    def extract_parameters(self):
        self.extract_config_parameters()
        self.extract_dataset_parameters()
        self.extract_threshold_parameters()
        self.extract_plot_parameters()
        self.extract_extra_parameters()
        self.extract_csv_parameters()
        print("Config parameters: ", self.configParameters)
        print("Dataset path: ", self.dataset_path)
        print("Thresholds: ", self.y_t1, self.y_t2, self.x_t1)
        print("Threshold: ", self.threshold)
        print("Plot parameters: ", self.plot_parameters)
        print("Save dir: ", self.save_dir)