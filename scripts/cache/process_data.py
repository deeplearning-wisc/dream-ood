import os
import shutil
import argparse

def move_images_classwise(classname):
    index = 0
    source_folder = '/nobackup/my_xfdu/sd/txt2img-samples-cifar100/txt2img-samples_select_50_sigma_0.07/'
    for ids, dirs in enumerate(os.listdir(source_folder)):
        dir_name = os.path.join(source_folder, dirs)
        for ids, dirs1 in enumerate(os.listdir(dir_name+'/samples')):
            # breakpoint()
            if classname in dirs1:
                shutil.copy(os.path.join(dir_name+'/samples', dirs1), os.path.join(
                    '/nobackup/my_xfdu/sd/txt2img-samples-cifar100/new_folder/' + classname, str(index) + '.png'))
                index += 1

move_images_classwise('beaver')