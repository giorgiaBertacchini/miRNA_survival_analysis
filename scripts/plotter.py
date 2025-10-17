from matplotlib import pyplot as plt
import numpy as np

class Plotter():
    def __init__(self):
        pass

    def plot_real_vs_predicted(self, trues, preds):
        trues_flat = trues.flatten()
        preds_flat = preds.flatten()
        errors = preds_flat - trues_flat
        
        plt.figure(figsize=(7,7))
        plt.scatter(trues_flat, preds_flat, c=np.abs(errors), cmap='viridis', alpha=0.7)
        plt.plot([trues.min(), trues.max()],
                [trues.min(), trues.max()],
                'r--', label='Perfect prediction')
        plt.colorbar(label="Absolute Error")
        plt.xlabel("Età reale")
        plt.ylabel("Età predetta")
        plt.title("Predicted vs True Age (colored by error)")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_losses_zoom(self ,train_losses, val_losses):
        x = np.arange(10, len(train_losses))
        plt.figure(figsize=(8,5))
        plt.plot(x, train_losses[10:], label="Train Loss")
        plt.plot(x, val_losses[10:], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training and Validation Loss over Epochs")
        plt.legend()
        plt.grid(True)    
        plt.show()
        
    def plot_losses_from_gridSearch(self, df_history):
        plt.figure(figsize=(8,5))
        plt.plot(df_history['epoch'][10:], df_history['train_loss'][10:], label='Train Loss')
        plt.plot(df_history['epoch'][10:], df_history['valid_loss'][10:], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()