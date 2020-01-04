import numpy as np
import sys, os

try:
    in_dir, feature_type = str(sys.argv[1]), str(sys.argv[2])
except IndexError:
    print('Format: python {} [in_dir] [feature_type]'.format(sys.argv[0]))
    sys.exit(1)

if feature_type != 'lpq' and feature_type != 'lbp':
    print('feature_type could be \'lpq\' or \'lbp\'')
    sys.exit(1)

out_dir = '{}'.format(feature_type)
if in_dir == out_dir:
    print('[in_dir] must differ from {}'.format(out_dir))
    sys.exit(1)
os.system('rm -fr {}'.format(out_dir))
os.mkdir(out_dir)

video_names = os.listdir(in_dir)
for video_name in video_names:
    if not os.path.isdir(os.path.join(in_dir,video_name)): continue
    in_path = os.path.join(in_dir, video_name, 'features_{}.npy'.format(feature_type))
    features = np.load(in_path)
    if features.shape[0] > 0:
        os.system('cp {} {}'.format(in_path, os.path.join(out_dir, '{}.npy'.format(video_name))))
sys.exit(0)
