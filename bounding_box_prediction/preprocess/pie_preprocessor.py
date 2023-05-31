import os
import sys
import argparse
import numpy as np
import pandas as pd
import pie_data


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Path to cloned PIE repository')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of train video between [0.1]')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of val video between [0.1]')
parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test video between [0.1]')

args = parser.parse_args()

data_path = args.data_path
sys.path.insert(1, data_path+'/')


if not os.path.isdir(os.path.join(data_path, 'processed_annotations')):
    os.mkdir(os.path.join(data_path, 'processed_annotations'))
    
if not os.path.isdir(os.path.join(data_path, 'processed_annotations', 'train')):    
    os.mkdir(os.path.join(data_path, 'processed_annotations', 'train'))

if not os.path.isdir(os.path.join(data_path, 'processed_annotations', 'val')):
    os.mkdir(os.path.join(data_path, 'processed_annotations', 'val'))

if not os.path.isdir(os.path.join(data_path, 'processed_annotations', 'test')):
    os.mkdir(os.path.join(data_path, 'processed_annotations', 'test'))

pie = pie_data.PIE(data_path=data_path)
dataset = pie.generate_database()

n_train_video = int(args.train_ratio * 346)
n_val_video = int(args.val_ratio * 346)
n_test_video = int(args.test_ratio * 346)

videos = list(dataset.keys())
train_indexes = [0,1,2,4]
train_videos = [videos[i] for i in train_indexes]
val_videos = videos[5]
test_videos = videos[3]

video_name = 0
for set_id in dataset:
    for video in dataset[set_id]:
        video_name += 1
        print('Processing', video, '...')
        vid = dataset[set_id][video]
        data = np.empty((0,12))
        for ped in vid['ped_annotations']:
            if vid['ped_annotations'][ped]['behavior']:
                frames = np.array(vid['ped_annotations'][ped]['frames']).reshape(-1,1)
                ids = np.repeat(ped, frames.shape[0]).reshape(-1,1)
                bbox = np.array(vid['ped_annotations'][ped]['bbox'])
                x = bbox[:,0].reshape(-1,1)
                y = bbox[:,1].reshape(-1,1)
                w = np.abs(bbox[:,0] - bbox[:,2]).reshape(-1,1)
                h = np.abs(bbox[:,1] - bbox[:,3]).reshape(-1,1)
                scenefolderpath = np.repeat(os.path.join(data_path, 'scene', video.replace('video_', '')), frames.shape[0]).reshape(-1,1)
                action = np.array(vid['ped_annotations'][ped]['behavior']['action']).reshape(-1,1)
                gesture = np.array(vid['ped_annotations'][ped]['behavior']['gesture']).reshape(-1,1)
                look = np.array(vid['ped_annotations'][ped]['behavior']['look']).reshape(-1,1)
                num_lanes = np.repeat(vid['ped_annotations'][ped]['attributes']['num_lanes'], frames.shape[0]).reshape(-1,1)

                #cross = np.array(vid['ped_annotations'][ped]['behavior']['cross']).reshape(-1,1)
                cross = np.array([])
                for i in vid['ped_annotations'][ped]['behavior']['cross']:
                    if i == 0 :
                        cross = np.append(cross, i)
                    elif i == 1 :
                        cross = np.append(cross, i)
                    else :
                        cross = np.append(cross, 0)

                cross = cross.reshape(-1,1)

                ped_data = np.hstack((frames, ids, x, y, w, h, scenefolderpath, cross, action, gesture, look, num_lanes))
                data = np.vstack((data, ped_data))
        data_to_write = pd.DataFrame({'frame': data[:,0].reshape(-1), 
                                    'ID': data[:,1].reshape(-1), 
                                    'x': data[:,2].reshape(-1), 
                                    'y': data[:,3].reshape(-1), 
                                    'w': data[:,4].reshape(-1), 
                                    'h': data[:,5].reshape(-1), 
                                    'scenefolderpath': data[:,6].reshape(-1), 
                                    'crossing_true': data[:,7].reshape(-1),
                                    'action': data[:,8].reshape(-1),
                                    'gesture': data[:,8].reshape(-1),
                                    'look': data[:,8].reshape(-1),
                                    'num_lanes': data[:,8].reshape(-1)})
        data_to_write['filename'] = data_to_write.frame
        data_to_write.filename = data_to_write.filename.apply(lambda x: '%04d'%int(x)+'.png')
        
        if set_id in train_videos:
            data_to_write.to_csv(os.path.join(data_path, 'processed_annotations', 'train', 'video_'+f'{video_name:04d}'+'.csv'), index=False)
        elif set_id in val_videos: 
            data_to_write.to_csv(os.path.join(data_path, 'processed_annotations', 'val', 'video_'+f'{video_name:04d}'+'.csv'), index=False)
        elif set_id in test_videos:
            data_to_write.to_csv(os.path.join(data_path, 'processed_annotations', 'test', 'video_'+f'{video_name:04d}'+'.csv'), index=False)