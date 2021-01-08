# std libs
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

# torch libs
import torch


def load_model(model, pretrained_path):
    """
    Loading pre-trained weights of a model
    INPUTS:
        model -> a pytorch model which will be updated with the pre-trained weights 
        pretrained_path -> path to where the .pth file is saved 
    
    RETURN:
        the updated model 
    """
    weights = torch.load(pretrained_path)
    pretrained_dict = weights['Saved_Model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    

def save_checkpoint_classifier(model, epoch, iteration, prefix="", dir_path = None):
    """
    Saving pre-trained model for inference 
    
    INPUTS:
        model-> PT model which we want to save
        epoch-> the current epoch number (will be used in the filename)
        iteration -> current iteration (will be used in the filename)
        prefix (optional)-> a prefix to the filename 
        dir_path (optional)-> path to save the pre-trained model
    
    """
        
    if not dir_path:
        dir_path = "./ClassifierWeights/"
        
    model_out_path = dir_path + prefix +f"model_epoch_{epoch}_iter_{iteration}.pth"
    state = {"epoch": epoch ,"Saved_Model": model}
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(state, model_out_path)
    print(f"Classifier Checkpoint saved to {model_out_path}")
        
        
        
def evaluate_classifier(valid_data_loader, cf_model):
    """
    Evaluating the performance of the network on validation/test dataset
    
    INPUTS:
        valid_data_loader-> a dataloader of the validation or test dataset 
        cf_model-> the model which we want to use for validation
    
    RETURN:
        None
    
    """
    print("==> Evaluating on Validation Set:")
    total = 0;
    correct = 0;
        
    with torch.no_grad():
        for _, batch in enumerate(valid_data_loader,0):
            data = batch[0].cuda()
            labels = batch[1].cuda()
            outputs = cf_model(data)
            _, predicted = torch.max(outputs.squeeze(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'    -> Accuracy of classifier network on validation set: {(100 * correct / total):4.4f} %' )
    # calculating the precision/recall based multi-label F1 score
    macro_score = f1_score(labels.detach().cpu().numpy(),predicted.detach().cpu().numpy(), average = 'macro' )
    w_score = f1_score(labels.detach().cpu().numpy(),predicted.detach().cpu().numpy(), average = 'weighted' )
    print(f'    -> Non-Weighted F1 Score on validation set: {macro_score:4.4f} ' )
    print(f'    -> Weighted F1 Score on validation set: {w_score:4.4f} ' )