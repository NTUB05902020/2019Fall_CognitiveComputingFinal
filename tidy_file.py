import numpy as np
import sys, os
import shutil

try:
    in_dir, feature_type = str(sys.argv[1]), str(sys.argv[2])
except IndexError:
    print('Format: python {} [in_dir] [feature_type]'.format(sys.argv[0]))
    sys.exit(1)

if feature_type != 'lpq' and feature_type != 'lbp' and feature_type != 'lpq_top':
    print('feature_type could be \'lpq\', \'lbp\' or lpq_top')
    sys.exit(1)

out_dir = '{}'.format(feature_type)
if in_dir == out_dir:
    print('[in_dir] must differ from {}'.format(out_dir))
    sys.exit(1)
try:
    shutil.rmtree(out_dir)
except FileNotFoundError:
    haha = 0
    ##

os.mkdir(out_dir)

video_names = os.listdir(in_dir)
for video_name in video_names:
    feature_dir = os.path.join(in_dir,video_name)
    if os.path.isdir(os.path.join(feature_dir, 'aligned')):
        feature_dir = os.path.join(feature_dir, 'aligned')
        
    in_path = os.path.join(feature_dir, 'features_{}.npy'.format(feature_type))
    
    try:
        features = np.load(in_path)
        if features.shape[0] > 0:
            out_path = os.path.join(out_dir, '{}.npy'.format(video_name))
            os.rename(in_path, out_path)
            print("Successfully moved {} to {}".format(in_path, out_path))
        else:
            print("No feature in FILE {}".format(in_path))
    except FileNotFoundError:
        print("FILE NOT FOUND IN {}".format(in_path))
    except ValueError:
        print("IGNORE feature in {}".format(in_path))

sys.exit(0)
