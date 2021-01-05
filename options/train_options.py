from .base_options import BaseOptions
import yaml

class TrainOptions(BaseOptions):

    def __init__(self, config_file):
        BaseOptions.__init__(self, config_file)
        self.initialize_train()

    def initialize_train(self):
        with open(self.filename) as file:
            config = yaml.safe_load(file)
        
        self.print_freq = int(config['training']['print_freq'])
        self.n_epochs = int(config['training']['n_epochs'])
        self.vgg_path = config['training']['vgg_path']
        self.lr = float(config['training']['lr'])
        self.itg = int(config['training']['itg'])
        self.itd = int(config['training']['itd'])
        self.beta1 = float(config['training']['beta1'])
        self.lmse = float(config['training']['lmse'])
        self.lssim = float(config['training']['lssim'])
        self.ladv = float(config['training']['ladv'])
        self.lgrad = float(config['training']['lgrad'])
        self.xtrain = config['dataset']['xtrain']
        self.ytrain = config['dataset']['ytrain']
        self.xtest = config['dataset']['xtest']
        self.ytest = config['dataset']['ytest']
        self.filename = config['dataset']['filename']
        self.lr_policy = config['training']['lr_policy']
        self.continue_train = config['training']['continue_train']
        self.source_distance = config['setup']['source_distance']
        self.det_distance = config['setup']['det_distance']
        self.det_count = config['setup']['det_count']
        self.det_spacing = config['setup']['det_spacing']
        self.parallel = config['setup']['parallel']
        self.isTrain = True
