import torch.nn as nn

def deleteEncodingLayers(model, num_layers_to_keep):
    oldModuleList = model.wav2vec2.encoder.layers
    newModuleList = nn.ModuleList()

    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    model.wav2vec2.encoder.layers = newModuleList
    return model
