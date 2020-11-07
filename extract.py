import torch as th
import os
import math
import numpy as np
import pandas as pd
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F
import subprocess
import cv2
import ntpath


parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--csv',
    type=str,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--type', type=str, default='2d',
                    help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,
                    help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                    help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                    help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                    help='Resnext model path')
args = parser.parse_args()

dataset = VideoLoader(
    args.csv,
    framerate=1 if args.type == '2d' else 24,
    size=224 if args.type == '2d' else 112,
    centercrop=(args.type == '3d'),
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing(args.type)
model = get_model(args)


def path_leaf(path: str):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_number_of_stream(filename):
    cap = cv2.VideoCapture('/content/test.mp4')
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        # cv2.imwrite('kang'+str(i)+'.jpg', frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return i


def create_data_frame(df, file_name):
    directory = os.path.dirname(file_name)
    new_file_name = directory+"/"+path_leaf(file_name)+"_duration_frame.csv"

    pd.DataFrame(df).to_csv(new_file_name, index=False,
                            header=["name", "video_frame_nbr", "video_length"])


df = []
with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data['input'][0]
        output_file = data['output'][0]
        input_file_length = get_length(input_file)
        input_file_stream_number = get_number_of_stream(input_file)
        df.append([path_leaf(input_file),
                   input_file_stream_number, input_file_length])
        if len(data['video'].shape) > 3:
            print('Computing features of video {}/{}: {}'.format(
                k + 1, n_dataset, input_file))
            video = data['video'].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype('float16')
                np.save(output_file, features)
        else:
            print('Video {} already processed.'.format(input_file))

create_data_frame(df, args.csv)