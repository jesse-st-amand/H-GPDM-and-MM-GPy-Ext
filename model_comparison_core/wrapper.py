from model_comparison_core.architectures.structures import TransformerMotionModel, VAEMotionModel, SequenceClassifier
from HGPLVM.hgp import Model_Wrapper
import torch.nn as nn
import torch.optim as optim
from HGPLVM.architectures.factory_functions import rnn_arch_factory
class TransformerModelWrapper(Model_Wrapper):
    def setup(self):
        criterion_classification = nn.CrossEntropyLoss()
        criterion_generation = nn.MSELoss()
        optimizer = optim.Adam(self.arch.parameters(), lr=0.001)
        return criterion_classification, criterion_generation, optimizer

    def build_architecture(self, data_set_class):
        if self.arch_dict is None:
            raise ValueError('No architecture set.')
        self.arch = TransformerMotionModel(self, data_set_class,
            sample_len=self.arch_dict['sample_len'],
            input_size=self.arch_dict['input_size'],
            num_classes=self.arch_dict['num_classes'],
            hidden_size_multiplier=self.arch_dict.get('hidden_size_multiplier', 768),
            num_layers=self.arch_dict.get('num_layers', 6),
            num_heads=self.arch_dict.get('num_heads', 12),
            dropout=self.arch_dict.get('dropout', 0.1),
            max_seq_length=self.arch_dict.get('max_seq_length', 100)
        )
        self.arch.data_set_class = data_set_class

    def optimize(self, *args, **kwargs):
        self.arch.optimize(*args, **kwargs)

    def IF_setup(self):
        return self.arch.IF_setup()



class RNN_Wrapper(Model_Wrapper):
    '''def __init__(self, model_dict, data_set_class = None):
        super().__init__(model_dict, data_set_class)'''
    def setup(self):
        criterion_classification = nn.CrossEntropyLoss()
        criterion_generation = nn.MSELoss()
        optimizer = optim.Adam(self.arch.parameters(), lr=0.001)
        return criterion_classification, criterion_generation, optimizer

    def build_architecture(self,data_set_class):
        if self.arch_dict is None:
            raise ValueError('No architecture set.')
        self.arch = rnn_arch_factory(self, data_set_class, self.arch_dict)

        if self.arch_dict is None:
            raise ValueError('No architecture set.')
        self.arch = SequenceClassifier(self, data_set_class,
            sample_len=self.arch_dict['sample_len'],
            input_size=self.arch_dict['input_size'],
            num_classes=self.arch_dict['num_classes'],
            hidden_size=self.arch_dict.get('hidden_size', 256),
            num_layers=self.arch_dict.get('num_layers', 32)
        )
        self.arch.data_set_class = data_set_class

    def optimize(self, *args, **kwargs):
        self.arch.optimize(*args, **kwargs)

    def IF_setup(self):
        return self.arch.IF_setup()

class VAEModelWrapper(Model_Wrapper):
    def setup(self):
        optimizer = optim.Adam(self.arch.parameters(), lr=0.001)
        return optimizer

    def build_architecture(self, data_set_class):
        if self.arch_dict is None:
            raise ValueError('No architecture set.')
        self.arch = VAEMotionModel(self, data_set_class,
            sample_len=self.arch_dict['sample_len'],
            input_size=self.arch_dict['input_size'],
            num_classes=self.arch_dict['num_classes'],
            hidden_size=self.arch_dict.get('hidden_size', 256),
            latent_size=self.arch_dict.get('latent_size', 32)
        )
        self.arch.data_set_class = data_set_class

    def optimize(self, num_epochs):
        optimizer = self.setup()
        self.arch.optimize(optimizer, num_epochs=num_epochs)

    def IF_setup(self):
        return self.arch.IF_setup()