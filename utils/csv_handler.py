import csv
import os

class CSVHandler:
    def __init__(self, path, csv_file_name) -> None:
        self.path = path
        self.csv_file_name = csv_file_name
        self.csv_file_path = os.path.join(self.path, self.csv_file_name)
        self.csv_writer = None
    
    def init_csv_writer(self):
        self.csv_writer = csv.writer(open(self.csv_file_path, 'w', newline=''))
    
    def write_csv_row(self, row):
        self.csv_writer.writerow(row)
    
    def close_csv_writer(self):
        self.csv_writer.close()
        self.csv_writer = None
