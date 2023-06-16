from src.posenet import model


def get_posenet(cfg):
    PoseNet_freq = cfg['posenet']['PoseNet_freq']
    layers_feat = cfg['posenet']['layers_feat']
    min_time = cfg['posenet']['min_time']
    max_time = cfg['posenet']['max_time']

    posenet = model.PoseNet()

    return posenet
    