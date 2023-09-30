import os
from collections import OrderedDict
import shutil
def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames

classnames = read_classnames('./scripts/classnames.txt')
f1 = ['n01498041', 'n01514859', 'n01582220', 'n01608432', 'n01616318',
          'n01687978', 'n01776313', 'n01806567', 'n01833805', 'n01882714',
          'n01910747', 'n01944390', 'n01985128', 'n02007558', 'n02071294',
          'n02085620', 'n02114855', 'n02123045', 'n02128385', 'n02129165',
          'n02129604', 'n02165456', 'n02190166', 'n02219486', 'n02226429',
          'n02279972', 'n02317335', 'n02326432', 'n02342885', 'n02363005',
          'n02391049', 'n02395406', 'n02403003', 'n02422699', 'n02442845',
          'n02444819', 'n02480855', 'n02510455', 'n02640242', 'n02672831',
          'n02687172', 'n02701002', 'n02730930', 'n02769748', 'n02782093',
          'n02787622', 'n02793495', 'n02799071', 'n02802426', 'n02814860',
          'n02840245', 'n02906734', 'n02948072', 'n02980441', 'n02999410',
          'n03014705', 'n03028079', 'n03032252', 'n03125729', 'n03160309',
          'n03179701', 'n03220513', 'n03249569', 'n03291819', 'n03384352',
          'n03388043', 'n03450230', 'n03481172', 'n03594734', 'n03594945',
          'n03627232', 'n03642806', 'n03649909', 'n03661043', 'n03676483',
          'n03724870', 'n03733281', 'n03759954', 'n03761084', 'n03773504',
          'n03804744', 'n03916031', 'n03938244', 'n04004767', 'n04026417',
          'n04090263', 'n04133789', 'n04153751', 'n04296562', 'n04330267',
          'n04371774', 'n04404412', 'n04465501', 'n04485082', 'n04507155',
          'n04536866', 'n04579432', 'n04606251', 'n07714990', 'n07745940']


# for imagenet-v2
name_list = list(classnames.keys())
copy_name = 'a'
for name in f1:
    index = name_list.index(name)
    shutil.copytree(os.path.join(
        '/nobackup-slow/dataset/my_xfdu/imagenetv2/imagenetv2-matched-frequency-format-val/', str(index)),
        os.path.join(
            '/nobackup-slow/dataset/my_xfdu/imagenetv2/processed/', copy_name)
    )
    copy_name += 'a'

# for imageneta
name_list = os.listdir('/nobackup-slow/dataset/my_xfdu/imageneta/imagenet-a/')
index = 0
for name in f1:
    # index = name_list.index(name)
    if name in name_list:
        shutil.copytree(os.path.join(
            '/nobackup-slow/dataset/my_xfdu/imageneta/imagenet-a', name),
            os.path.join(
                '/nobackup-slow/dataset/my_xfdu/imageneta/processed/', str(index))
        )
    index += 1

