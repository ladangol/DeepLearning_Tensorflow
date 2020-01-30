import config
from util import get_path
import simple_cnn
import cam
from inception_resnet_v2 import model as inception_resnet_v2
from train import grid_search
from train import train

class GenerateModel:
    def __init__(self,model_type):
        self.fActive = self.generate_model(model_type) # mapping: string --> variable = function name

    def generate_model(self, model_type):
        switcher = {
            's':   simple_cnn.define_model,
            'c':   cam.define_model,
            'ir2': inception_resnet_v2.define_model
        }

        current_model = switcher.get(model_type.lower(), None)

        if current_model == None:
            raise AssertionError("model_type is invalid!")

        return current_model

    def run_function(self, config):
        return  self.fActive(config)

class GenerateTrainer:
    def __init__(self,training_type):
        self.fActive = self.generate_trainer(training_type) # mapping: string --> variable = function name

    def generate_trainer(self, training_type):
        switcher = {
            'gs': grid_search,
            't': train
        }

        trainer = switcher.get(training_type.lower(), None)

        if trainer == None:
            raise AssertionError("training_type is invalid!")

        return trainer

    def run_function(self, in_model_generater, in_config, training_data):
        self.fActive(in_model_generater, in_config, training_data)

class GeneratePredicter:
    def __init__(self, predicting_type):
        self.fActive = self.generate_predicter(predicting_type) # mapping: string --> variable = function name

    def generate_predicter(self, predicting_type):
        switcher = {
            'p': self.predict,
            'c': self.cam_predict
        }

        predicter = switcher.get(predicting_type.lower(), None)

        if predicter == None:
            raise AssertionError("predicting_type is invalid!")

        return predicter

    def predict(self):
            test_model_path = get_path(config.model_path_root, 'no_cam\\Model-60-0.820.model')
            test_data_path = get_path(config.data_path_root, 'test')
            simple_cnn.predict(test_data_path, test_model_path, config)

    def cam_predict(self, config):
        test_model_path = get_path(config.model_path_root, 'Vgg_16_Cam\\Model-02-0.978.model')
        test_data_path = get_path(config.data_path_root, 'test\\cam')
        cam.predict(test_data_path, test_model_path, config.image_size)

    def run_function(self, config):
        self.fActive(config)



