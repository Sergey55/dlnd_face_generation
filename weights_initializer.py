from torch.nn import init

class WeightsInitializer():
    def init_weights_normal(self, model):
        """
        Applies initial weights to certain layers in a model.
        
        The weights are taken from a normal distribution with 
        mean = 0 and std = 0.02

        Args:
            model       A module or layer in a network
        """
        def init_func(m):
            classname = m.__class__.__name__

            if (hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1)):
                init.normal_(m.weight.data, 0.0, 0.02)

        model.apply(init_func)

    def init_weights_kaiming(self, model):
        """
        Applies initial weights to a certain layers in a model.

        Args:
            model       A module or layer in a network
        """
        def init_func(m):
            classname = m.__class__.__name__

            if (hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1)):
                init.kaiming_normal_(tensor=m.weight.data, a=0.2, nonlinearity='leaky_relu')

        model.apply(init_func)

    def init_weights_xavier(self, model):
        """
        Applies initial weights to certain layers in a model.

        Args:
            model       A module or layer in a network
        """
        def init_func(m):
            classname = m.__class__.__name__

            if (hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1)):
                init.xavier_normal_(tensor=m.weight.data)

        model.apply(init_func)