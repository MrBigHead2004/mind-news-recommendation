import torch

def save_model(model, optimizer, history, config, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config,
    }, save_path)