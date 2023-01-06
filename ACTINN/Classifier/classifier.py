import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__(self, output_dim=None, input_size=None):
        """

        The Classifer class: We are developing a model similar to ACTINN for good accuracy

        """
        if output_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input dim (num features) and output dim (number of classes)')

        super().__init__();
        self.inp_dim = input_size;
        self.out_dim = output_dim;

        # feed forward layers
        self.classifier_sequential = nn.Sequential(
                                        nn.Linear(self.inp_dim, 100),
                                        nn.ReLU(),

                                        nn.Linear(100, 50),
                                        nn.ReLU(),

                                        nn.Linear(50, 25),
                                        nn.ReLU(),

                                        nn.Linear(25, output_dim)
                                        # SoftMax not needed for CrossEntropyLoss in PyTorch
                                        ## i.e. no need for nn.Softmax(dim=1)
                                        )

    def forward(self, x, **kwargs):
        """

        Forward pass of the classifier

        """
        out = self.classifier_sequential(x);

        return out
    
    

    
class Attn_Classifier(nn.Module):

    def __init__(self, output_dim=None, input_size=None, **kwargs):
        """

        The Classifer class: We are developing a model similar to ACTINN for good accuracy

        """
        if output_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input dim (num features) and output dim (number of classes)')

        super().__init__();
        self.inp_dim = input_size;
        self.out_dim = output_dim;

        self.attn_layer = nn.Linear(self.inp_dim, self.inp_dim)
        
        # feed forward layers
        self.classifier_sequential = nn.Sequential(
                                        nn.Linear(self.inp_dim, 100),
                                        nn.ReLU(),

                                        nn.Linear(100, 50),
                                        nn.ReLU(),

                                        nn.Linear(50, 25),
                                        nn.ReLU(),

                                        nn.Linear(25, output_dim)
                                        # SoftMax not needed for CrossEntropyLoss in PyTorch
                                        ## i.e. no need for nn.Softmax(dim=1)
                                        )

    def forward(self, x, training=True, **kwargs):
        """

        Forward pass of the classifier

        """
        # attention
        alpha = self.softmax(self.attn_layer(x));
        x_c = self.context(alpha, x)
        
        out = self.classifier_sequential(x_c);

        return out


    def softmax(self, e_t):
        """
        Step 3:
        Compute the probabilities alpha_t
        In : torch.Size([batch_size, sequence_length, 1])
        Out: torch.Size([batch_size, sequence_length, 1])
        """
        #### SparseMax
        # sparsemax = Sparsemax(dim=1)
        # alphas = sparsemax(e_t)

        #### SoftMax
        softmax = torch.nn.Softmax(dim=1)
        alphas = softmax(e_t)
        return alphas
    

    def context(self, alpha_t, x_t):
        """
        Step 4:
        Compute the context vector c
        In : torch.Size([batch_size, sequence_length, 1]), torch.Size([batch_size, sequence_length, sequence_dim])
        Out: torch.Size([batch_size, 1, hidden_dimensions])
        """
        return torch.mul(alpha_t, x_t)
    
    
class TransferLearning(nn.Module):

    def __init__(self, pretrained_model, output_dim=None, input_size=None):
        """

        The Classifer class: We are developing a model similar to ACTINN for good accuracy

        """
        if output_dim == None or input_size == None:
            raise ValueError('Must explicitly declare input dim (num features) and output dim (number of classes)')

        super().__init__();
        self.inp_dim = input_size;
        self.out_dim = output_dim;

        #----- the pretrained layers that we will not re-train
        self.classifier_sequential = pretrained_model.classifier_sequential[0:-1]
        
        #---- last layer that we will train
        self.prediction_module = nn.Linear(25, output_dim)

    def forward(self, x, **kwargs):
        """

        Forward pass of the classifier

        """
        # freeze the pre-trained layers
        with torch.no_grad():
            x_frozen = self.classifier_sequential(x);
        
        predictions = self.prediction_module(x_frozen)

        return predictions
    
