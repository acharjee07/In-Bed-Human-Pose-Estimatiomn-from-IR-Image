
import sys
import os
sys.path.insert(0,os.path.abspath(os.getcwd()))

import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import math
import torchvision.transforms as transforms
from lib.utils import HeatmapProcessor, heatmap2keypts, heatmap2keyptsBatch
from lib.utils import calcAllPCKhBatch
import random
current_dir=os.getcwd()


import pickle
def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)





def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds








def get_pck_single(gt,pred,th):
    head_size=np.linalg.norm(gt[12]-gt[13])
    distances=np.linalg.norm(gt-pred,axis=1)
    pck=np.sum(distances<head_size*.5)/14
    return pck
def pck_on_batch(gt_batch,pred_batch):
    batch_pck=[]
    for x in range(len(gt_batch)):
        gt=gt_batch[x]
        pred=pred_batch[x]
        batch_pck.append(get_pck_single(gt,pred,th=.5))
    return batch_pck












def joints_mse_loss(output, target, target_weight=None):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.view((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.view((batch_size, num_joints, -1)).split(1, 1)

    loss = 0
    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx]
        heatmap_gt = heatmaps_gt[idx]
#         if target_weight is None:
        loss += 0.5 * mse_loss(heatmap_pred, heatmap_gt, reduction='mean')
#         else:
#             loss += 0.5 * mse_loss(
#                 heatmap_pred.mul(target_weight[:, idx]),
#                 heatmap_gt.mul(target_weight[:, idx]),
#                 reduction='mean'
#             )

    return loss / num_joints


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        if not self.use_target_weight:
            target_weight = None
        return joints_mse_loss(output, target, target_weight)

    



from torch.utils.data import DataLoader,ConcatDataset
import pytorch_lightning as pl
from lib.utils import SLPdataset
from lib.utils import MPIIDataset
from lib.dataset import SLPDatasetJointToLabels, SLPDatasetLeftRightJointPairs, loadImagePathsAndLabels
import sklearn 
from sklearn.metrics import auc

import os
# os.path.join('.','data', 'train','train', '*')
# validImgPaths, validKeyPts = loadImagePathsAndLabels(os.path.join('.','current_dir','data', 'valid','valid'), onlyAnnotated=True)
# trainImgPaths, trainKeyPts = loadImagePathsAndLabels(os.path.join('.','current_dir','data', 'train','train'), onlyAnnotated=False)
# annotatedImgPaths=trainImgPaths[0:1350]
# unannotatedImgPaths=trainImgPaths[1350:]



class LitPose(pl.LightningModule):
    def __init__(self,plConfig,data_config, model,phase,preds,refined_path):
        super(LitPose, self).__init__()
        self.model = model
        self.plConfig=plConfig
        self.sigmoid = nn.Sigmoid()
        self.lr = self.plConfig.lr
        self.data_config=data_config

        self.validImgPaths, self.validKeyPts = loadImagePathsAndLabels(self.data_config['slp_valid_path'], onlyAnnotated=True)
        self.trainImgPaths, self.trainKeyPts = loadImagePathsAndLabels(self.data_config['slp_train_path'], onlyAnnotated=False)
        print('dfdlfkjlskjdfljaldjflakjdfajoijlj')
        print(len(self.trainImgPaths))
        self.annotatedImgPaths=self.trainImgPaths[0:1350]
        self.unannotatedImgPaths=self.trainImgPaths[1350:]
        


        self.phase=phase
        if self.phase==2:
            self.predictions=preds
#             self.selected_path=list(self.predictions.keys())
            self.selected_path=refined_path
            

        
        
        
        
    def forward(self, x, *args, **kwargs):
        return self.model(x)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.plConfig.lr, betas=self.plConfig.betas, eps=self.plConfig.eps,
                                            weight_decay=self.plConfig.weight_decay, amsgrad=self.plConfig.amsgrad)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             epochs=self.plConfig.n_epoch, steps_per_epoch=self.plConfig.steps_per_epoch,
                                                             max_lr=self.plConfig.max_lr, pct_start=self.plConfig.pct_start, 
                                                   div_factor=self.plConfig.div_factor, final_div_factor=self.plConfig.final_div_factor)
        scheduler = {'scheduler': self.scheduler, 'interval': 'step',}
        return [self.optimizer], [scheduler]


    def train_dataloader(self):
        if self.phase==0:
            
            img_root = self.data_config['mpii_images']
            anno_path = self.data_config['mpii_train_json']
            img_size = self.data_config['input_size'][1:]
            hm_size = (self.data_config['hm_size'][0],self.data_config['hm_size'][1])
            transform= transforms.Compose([ transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                            ])
            is_train = True
            trainDataset = MPIIDataset(img_root, anno_path, img_size, 
                                       hm_size, is_train, transform=transform)






        elif self.phase==1:
            trainDataset = SLPdataset(self.data_config,
                         self.annotatedImgPaths[0:10],
                         self.trainKeyPts,
                         outputHeatmap=True,
                        heatmapRes=(self.data_config['hm_size'][0],self.data_config['hm_size'][1]),
                        
                         normalizeImg=True,
                         normalizeKeyPts=True,
                         shuffle=True,
                         leftRightJointPairIndexes=SLPDatasetLeftRightJointPairs,
                         probFlipH=.5,
                         probMaskRandom=.5,
                         probGaussianNoise=.5,
                         probAttu=1,
                         resize=True,
                         epoch=(30 ,self.current_epoch)
                         
                         )
        
        
        
        elif self.phase==2:
            

            random.shuffle(self.selected_path)
            

            trainDataset1=SLPdataset(self.data_config,
                         self.annotatedImgPaths,
                         self.trainKeyPts,
                         outputHeatmap=True,
                         heatmapRes=(self.data_config['hm_size'][0],self.data_config['hm_size'][1]),
                        
                         normalizeImg=True,
                         normalizeKeyPts=True,
                         shuffle=True,
                         leftRightJointPairIndexes=SLPDatasetLeftRightJointPairs,
                         probFlipH=.5,
                         probMaskRandom=.5,
                         probGaussianNoise=.5,
                         probAttu=1,
                     
                          resize=True,
                          epoch=(30 ,self.current_epoch)
                         
                         )
            trainDataset2=SLPdataset(self.data_config,
                         self.selected_path,
                         self.predictions,
                         outputHeatmap=True,
                         heatmapRes=(self.data_config['hm_size'][0],self.data_config['hm_size'][1]),
                        
                         normalizeImg=True,
                         normalizeKeyPts=True,
                         shuffle=True,
                         leftRightJointPairIndexes=SLPDatasetLeftRightJointPairs,
                         probFlipH=.5,
                         probMaskRandom=0,
                         probGaussianNoise=.1,
                         probAttu=0,
                     
                          resize=True,
                          epoch=None
                         
                         )
            trainDataset=ConcatDataset([trainDataset1,trainDataset2])   
        else:
            print('Phase must be of 1 or 2')
        
        
        
        
             
        train_loader = DataLoader(trainDataset, batch_size=self.data_config['batch_size'], shuffle=True, pin_memory=True, drop_last=True, num_workers=2)
        return train_loader
    
    def val_dataloader(self):

        if self.phase==0:
            
            img_root = self.data_config['mpii_images']
            anno_path = self.data_config['mpii_valid_json']
            img_size = self.data_config['input_size'][1:]
            hm_size = (self.data_config['hm_size'][0],self.data_config['hm_size'][1])
            transform= transforms.Compose([ transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                            ])
            is_train = True
            validDataset = MPIIDataset(img_root, anno_path, img_size, 
                                       hm_size, is_train, transform=transform)
        else:

            validDataset = SLPdataset(self.data_config,self.validImgPaths[0:5], self.validKeyPts,
                            outputHeatmap=True,  heatmapRes=(self.data_config['hm_size'][0],self.data_config['hm_size'][1]),
                            normalizeImg=True, normalizeKeyPts=True, shuffle=False,probAttu=0,resize=True)      
        valid_loader = DataLoader(validDataset, batch_size=self.data_config['batch_size'], shuffle=False, pin_memory=False, drop_last=True, num_workers=2)
        return valid_loader
    
 



    
    def training_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = joints_mse_loss(output, target)
        score,_=calcAllPCKhBatch(get_preds(target.cpu()),get_preds(output.detach().cpu()),th=.5)
        
        # X=np.arange(0,.5,.1)
        # Y=[]
        # for x in X :
        #     y,_=calcAllPCKhBatch(get_preds(target.cpu()),get_preds(output.detach().cpu()),th=x)
        #     Y.append(y)
            
        # sauc=auc(X,Y)

        target=get_preds(target.cpu())
        output=get_preds(output.detach().cpu())
        logs={'train_loss':loss, 'train_acc':score*100,'lr':self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True


        )
        return {'loss':loss ,'target':target,'output': output}


    def training_epoch_end(self,training_step_outputs):
#         targets=[x['target'][0] for x in training_step_outputs]
#         outputs=[x['output'][0] for x in training_step_outputs]
        
        targets=[]
        outputs=[]
        for data in training_step_outputs:
            targets.extend(data['target'])
            outputs.extend(data['output'])
            
#         print(len(targets))
#         print(len(targets))
#         print(targets)
#         print(len(outputs))
#         print(outputs)
#         targets=training_step_outputs[0][1]
#         outputs=training_step_outputs[1][1]
        X=np.arange(0,.5,.01)
        Y=[]
        for x in X :
            y,_=calcAllPCKhBatch(targets,outputs,th=x)
            Y.append(y)

        sauc=auc(X,Y)
        self.log_dict({'train_auc':sauc/.5})


    
    def validation_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = joints_mse_loss(output, target)
        score,_=calcAllPCKhBatch(get_preds(target.cpu()),get_preds(output.detach().cpu()),th=.5)    
        
        logs={'valid_loss':loss, 'valid_acc':score*100 }
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
            
        )
        target=get_preds(target.cpu())
        output=get_preds(output.detach().cpu())
        return loss ,target,output

    def validation_epoch_end(self,validtion_step_outputs):
        targets=[]
        outputs=[]
        for _,t,o in validtion_step_outputs:
            targets.extend(t)
            
            outputs.extend(o)
            
                        
        
#         targets=[x[1] for x in validtion_step_outputs]
#         outputs=[x[2] for x in validtion_step_outputs]
      
#         print(len(targets))
        X=np.arange(0,.5,.01)
        Y=[]
        for x in X :
            y,_=calcAllPCKhBatch(targets,outputs,th=x)
            Y.append(y)

        sauc=auc(X,Y)
        self.log_dict({'valid_auc':sauc/.5})

