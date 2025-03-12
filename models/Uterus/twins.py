from .Twins_main.gvt import alt_gvt_small
import collections
def twins(num_classes=100):

    model = alt_gvt_small(pretrained=True, img_size=256,num_classes=num_classes, drop_path_rate=0.1)
    return  model

