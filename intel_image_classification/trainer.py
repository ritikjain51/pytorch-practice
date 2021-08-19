from tqdm import tqdm
import torch


class Trainer(object):
    def __init__(self, optimizer, criteria, epochs=10, scheduler=None):
        """
        This class will train the model based on the 
        - optimizer: Optimizer algorithm (object)
        - criteria: It is the loss function which will be used like CrossEntropyLoss, MSELoss etc.
        """
        self.optimizer = optimizer
        self.epochs = epochs
        self.criteria = criteria
        self.scheduler = scheduler
        
    
    def train_one_step(self, x, y):
        """
        Training on Single Step
        - Predict the output
        - Optimize the parameters
        """
        self.optimizer.zero_grad() # Initialization of Gredients to 0
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_one_epoch(self, data_loader):
        
        """
        This function will enable the epoch training and return the loss for the epoch
        """
        self.model.train() # Setting model in Training Mode
        total_loss = 0
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            loss = self.train_one_step(**data)
            total_loss += loss
        return total_loss / (idx + 1)
    
    def eval_one_epoch(self, data_loader):
        """
        This function will enable the epoch training and return the loss for the epoch
        """
        self.model.eval() # Setting model in Evaluation Mode
        total_loss = 0
        for idx, data in enumerate(data_loader):
            x, y = data["x"], data["y"]
            y_hat = self.model(x)
            loss = self.criteria(y_hat, y)
            total_loss += loss.item()
        return total_loss / (idx + 1)
    
    def save_model(self, model_path, **kwargs):
        print("Saving Model....")
        torch.save(self.model, model_path)
        print("Model is Saved...")
    
    def fit(self, model, train_loader, valid_loader=None, scheduler=None, **kwargs):
        """
        This function will start the model training. 
        """
        if next(model.parameters()).device != kwargs.get("device", "cpu"):
            model = model.to(kwargs.get("device", "cpu"))
        self.model = model
        valid_loss = None
        for epoch in tqdm(range(self.epochs)):
            loss = self.train_one_epoch(train_loader)
            if valid_loader:
                valid_loss = self.eval_one_epoch(valid_loader)
            if hasattr(self, "sechduler") and self.sechduler != None:
                self.scheduler.step()
            tqdm.write(f"Epoch: {epoch}, Training Loss: {loss}, Validation Loss: {valid_loss}")
            self.save_model(f'{kwargs.get("model_path")}_{epoch}.pt')
        return self.model
    