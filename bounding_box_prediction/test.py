import time
import os
import argparse

import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torchvision
# import torchvision.transforms as transforms
    
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, precision_score

# import DataLoader
import datasets
import network
import utils
from utils import data_loader
import pandas as pd

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PV-LSTM network')
    
    parser.add_argument('--data_dir', type=str,
                        help='Path to dataset',
                        required=True)
    parser.add_argument('--dataset', type=str, 
                        help='Datasets supported: jaad, jta, nuscenes, pie',
                        required=True)
    parser.add_argument('--out_dir', type=str, 
                        help='Path to save output',
                        required=True)  
    parser.add_argument('--task', type=str, 
                        help='Task the network is performing, choose between 2D_bounding_box-intention, \
                            3D_bounding_box, 3D_bounding_box-attribute',
                        required=True)
    parser.add_argument('--txt_name', type=str, 
                        help='Name of the output file where the results will be saved',
                        default='results.txt')
    
    # data configuration
    parser.add_argument('--input', type=int,
                        help='Input sequence length in frames',
                        required=True)
    parser.add_argument('--output', type=int, 
                        help='Output sequence length in frames',
                        required=True)
    parser.add_argument('--stride', type=int, 
                        help='Input and output sequence stride in frames',
                        required=True)  
    parser.add_argument('--skip', type=int, default=1)  
    parser.add_argument('--is_3D', type=bool, default=False) 

    # data loading / saving           
    parser.add_argument('--dtype', type=str, default='train')
    parser.add_argument("--from_file", type=bool, default=False)       
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--loader_workers', type=int, default=10)
    parser.add_argument('--loader_shuffle', type=bool, default=False)
    parser.add_argument('--pin_memory', type=bool, default=False)

    # training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--lr_scheduler', type=bool, default=False)

    # network
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--hardtanh_limit', type=int, default=100)

    args = parser.parse_args()

    return args


# For 2D datasets
def test_2d(args, net, test_loader):
    print('='*100)
    print('Testing ...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
    modelname = 'model_' + file + '.pkl'

    net.load_state_dict(torch.load(os.path.join(args.out_dir, args.log_name, modelname)))
    net.eval()
    input=args.input
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    val_s_scores   = []
    val_c_scores   = []

    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0
    avg_acc = 0
    avg_rec = 0
    avg_pre = 0
    mAP = 0

    avg_epoch_val_s_loss   = 0
    avg_epoch_val_c_loss   = 0

    counter=0
    state_preds = []
    state_targets = []
    intent_preds = []
    intent_targets = []
    bounding_box_preds = np.empty((0, input, 4))
    bounding_box_true = np.empty((0, input, 4))
    file = np.empty((0,100))
    start = time.time()
    data = []


    for idx, (obs_s, target_s, obs_p, target_p, target_c, label_c, behaviour, crossing_obs, filename) in enumerate(test_loader):
       
        counter+=1
        
        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_p    = obs_p.to(device='cuda')
        target_p = target_p.to(device='cuda')
        target_c = target_c.to(device='cuda')
        behaviour = behaviour.to(device='cuda')
        file2 = []
        with torch.no_grad():
            speed_preds, crossing_preds, intentions = net(speed=obs_s, pos=obs_p, bh = behaviour, average=True)
            speed_loss    = mse(speed_preds, target_s)/100

            crossing_loss = 0
            for i in range(target_c.shape[1]):
                crossing_loss += bce(crossing_preds[:,i], target_c[:,i])
            crossing_loss /= target_c.shape[1]

            avg_epoch_val_s_loss += float(speed_loss)
            avg_epoch_val_c_loss += float(crossing_loss)

            preds_p = utils.speed2pos(speed_preds, obs_p)

            preds_p2 = preds_p.cpu().numpy()
            target_p2 = target_p.cpu().numpy()
     
            file2.append(filename)
            filename = np.array(filename)
    
            bounding_box_preds = np.vstack((bounding_box_preds, preds_p2))             
            bounding_box_true = np.vstack((bounding_box_true, target_p2))
            #file = np.vstack((file, filename))   


            ade += float(utils.ADE(preds_p, target_p))
            fde += float(utils.FDE(preds_p, target_p))
            aiou += float(utils.AIOU(preds_p, target_p))
            fiou += float(utils.FIOU(preds_p, target_p))

            target_c = target_c[:,:,1].view(-1).cpu().numpy()
            crossing_preds = np.argmax(crossing_preds.view(-1,2).detach().cpu().numpy(), axis=1)

            label_c = label_c.view(-1).cpu().numpy()
            intentions = intentions.view(-1).detach().cpu().numpy()

            state_preds.extend(crossing_preds)
            state_targets.extend(target_c)
            intent_preds.extend(intentions)
            intent_targets.extend(label_c)

        avg_epoch_val_s_loss += float(speed_loss)
        avg_epoch_val_c_loss += float(crossing_loss)



    avg_epoch_val_s_loss /= counter
    avg_epoch_val_c_loss /= counter

    val_s_scores.append(avg_epoch_val_s_loss)
    val_c_scores.append(avg_epoch_val_c_loss)

    ade  /= counter
    fde  /= counter     
    aiou /= counter
    fiou /= counter

    avg_acc = accuracy_score(state_targets, state_preds)
    avg_rec = recall_score(state_targets, state_preds, average='binary', zero_division=1)
    avg_pre = precision_score(state_targets, state_preds, average='binary', zero_division=1)
    intent_acc = accuracy_score(intent_targets, intent_preds)
    #data.append([bounding_box_preds, bounding_box_true])

    df = pd.DataFrame({'bbox_preds': bounding_box_preds.reshape(-1, 1, input, 4).tolist(), 
                       'bbox_truth': bounding_box_true.reshape(-1, 1, input, 4).tolist()})
                       #'file_name': np.expand_dims(file, axis=0)})

    if args.save:
        print('\nSaving ...')
        file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
        if args.lr_scheduler:
            filename = 'bbox_results_' + file + '_scheduler.csv'
        else:
            filename = 'bbox_results_' + file + '.csv'

        df.to_csv(os.path.join(args.out_dir, args.log_name, filename), index=False)

    with open(args.txt_name, 'w') as f:
        print('vs: %.4f' % avg_epoch_val_s_loss, '| vc: %.4f' % avg_epoch_val_c_loss, '| ade: %.4f' % ade,
            '| fde: %.4f' % fde, '| aiou: %.4f' % aiou, '| fiou: %.4f' % fiou, '| state_acc: %.4f' % avg_acc,
            '| int_acc: %.4f' % intent_acc,
            '| t:%.4f' % (time.time() - start), file=f)

    print('vs: %.4f'% avg_epoch_val_s_loss, '| vc: %.4f'% avg_epoch_val_c_loss, '| ade: %.4f'% ade, 
        '| fde: %.4f'% fde, '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| state_acc: %.4f'% avg_acc, 
        '| int_acc: %.4f'% intent_acc, 
        '| t:%.4f'%(time.time()-start))

    # For 3D datasets
def test_3d(args, net, test_loader):
    print('='*100)
    print('Testing 3D dataset...')
    print('Task: ' + str(args.task))
    print('Learning rate: ' + str(args.lr))
    print('Number of epochs: ' + str(args.n_epochs))
    print('Hidden layer size: ' + str(args.hidden_size) + '\n')

    file = '{}_{}'.format(str(args.lr), str(args.hidden_size)) 
    modelname = 'model_' + file + '.pkl'

    net.load_state_dict(torch.load(os.path.join(args.out_dir, args.log_name, modelname)))
    net.eval()

    mse = nn.MSELoss()
    bce = nn.BCELoss()

    results = []
    ade  = 0
    fde  = 0
    aiou = 0
    fiou = 0

    avg_epoch_val_s_loss   = 0
    avg_epoch_val_a_loss   = 0

    counter=0

    start = time.time()

    for idx, values in enumerate(test_loader):
        counter += 1

        if 'attribute' in args.task:
            (obs_s, target_s, obs_p, target_p, target_a) = values
            target_a = target_a.to(device='cuda')
        else:
            (obs_s, target_s, obs_p, target_p) = values

        obs_s    = obs_s.to(device='cuda')
        target_s = target_s.to(device='cuda')
        obs_p    = obs_p.to(device='cuda')
        target_p = target_p.to(device='cuda')

        with torch.no_grad():
            if 'attribute' in args.task:
                speed_preds, attrib_preds = net(speed=obs_s, pos=obs_p, average=False)
                
                attrib_loss = 0
                for i in range(target_a.shape[1]):
                    attrib_loss += bce(attrib_preds[:,i], target_a[:,i])    
                attrib_loss /= target_a.shape[1]
                avg_epoch_val_s_loss += float(attrib_loss)
            else:
                speed_preds = net(speed=obs_s, pos=obs_p, average=False)[0]
            
            speed_loss  = mse(speed_preds, target_s)
            avg_epoch_val_s_loss += float(speed_loss)

            preds_p = utils.speed2pos(speed_preds, obs_p, args.is_3D)
            ade += float(utils.ADE(preds_p, target_p, args.is_3D))
            fde += float(utils.FDE(preds_p, target_p, args.is_3D))
            aiou += float(utils.AIOU(preds_p, target_p, args.is_3D))
            fiou += float(utils.FIOU(preds_p, target_p, args.is_3D))
            results.append(preds_p)

    avg_epoch_val_s_loss /= counter
    ade  /= counter
    fde  /= counter     
    aiou /= counter
    fiou /= counter

    if 'attribute' in args.task:
        avg_epoch_val_a_loss /= counter
        print('vs: %.7f'% avg_epoch_val_s_loss, '| va: %.7f'% avg_epoch_val_a_loss, '| ade: %.4f'% ade, '| fde: %.4f'% fde, 
        '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| t:%.4f'%(time.time()-start))
    else:
        print('vs: %.7f'% avg_epoch_val_s_loss, '| ade: %.4f'% ade, '| fde: %.4f'% fde, 
        '| aiou: %.4f'% aiou, '| fiou: %.4f'% fiou, '| t:%.4f'%(time.time()-start))


if __name__ == '__main__':
    args = parse_args()

    # create output dir
    if not args.log_name:
        args.log_name = '{}_{}_{}_{}'.format(args.dataset, str(args.input),\
                                str(args.output), str(args.stride)) 
    if not os.path.isdir(os.path.join(args.out_dir, args.log_name)):
        os.mkdir(os.path.join(args.out_dir, args.log_name))

    # select dataset
    if args.dataset == 'jaad':
        args.is_3D = False
    elif args.dataset == 'pie':
        args.is_3D = False
    elif args.dataset == 'jta':
        args.is_3D = True
    elif args.dataset == 'nuscenes':
        args.is_3D = True
    else:
        print('Unknown dataset entered! Please select from available datasets: jaad, jta, nuscenes, pie...')

    # load data
    test_set = eval('datasets.' + args.dataset)(
                data_dir=args.data_dir,
                out_dir=os.path.join(args.out_dir, args.log_name),
                dtype='test',
                input=args.input,
                output=args.output,
                stride=args.stride,
                skip=args.skip,
                task=args.task,
                from_file=args.from_file,
                save=args.save
                )

    test_loader = data_loader(args, test_set)

    # initiate network
    net = network.PV_LSTM(args).to(args.device)

    # benchmark network


    # training
    test_2d(args, net, test_loader)