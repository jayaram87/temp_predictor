import logging
import os

class Logger:
    def __init__(self, filename):
        self.filename = filename

    def logger(self, logtype, error1):
        if self.filename not in os.listdir():
            with open(os.path.join(os.getcwd(), self.filename), 'a+') as f:
                print(f.read())

        logging.basicConfig(filename=os.path.join(os.getcwd(), self.filename), level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s')
        if logtype == 'INFO':
            logging.info(error1)
        elif logtype == 'ERROR':
            logging.error(error1)
        logging.shutdown()