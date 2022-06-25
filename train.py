import torch
import matplotlib.pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
import cv2
import os


from lib.utils import HeatmapProcessor
from models import c_model

from models import mxy
from train_utils_lit import LitPose

import wandb
# wandb.login(key='b19924dbbd8814abfc6253cb43cb4f741cdd4f98')  ##logging in sanjay
# from pytorch_lightning.loggers import WandbLogger

import pickle
def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)


def normaa(image):
    img=image.copy()
    maxpixel=np.max(img)
    mean0=0.485*maxpixel
    mean1=0.456*maxpixel
    mean2=0.406*maxpixel
    std0=0.229*maxpixel
    std1=0.224*maxpixel
    std2=0.225*maxpixel
    img[:,:,0]=((img[:,:,0]-mean0)/std0)
    img[:,:,1]=((img[:,:,1]-mean1)/std1)
    img[:,:,2]=((img[:,:,2]-mean2)/std2)
    return img


def get_inspected(pred):
    '''function for generating predicion scores'''
    
    selected_path=list(pred.keys())

    refined_path=[]
    confs=[]
    
    for i in tqdm(range(2249)):
        img_path=selected_path[i]
        annotation=pred[img_path]
        heatmapProcessor = HeatmapProcessor(160,120,64,64,14, scaleAware=True)
        heatmap = torch.tensor(heatmapProcessor.keypts2heatmap(annotation.copy(), sigma=2.0).astype(np.float32)).permute(2,0,1).unsqueeze(0)
        image_1=cv2.imread(img_path)
        image_2=torch.tensor(cv2.resize(normaa(image_1),(256,256)).astype(np.float32)).permute(2,0,1).unsqueeze(0)
        pred = c_model(image_2.cuda(),heatmap.cuda())
        out=pred.squeeze().detach().cpu().numpy()
        confs.append(out)
        if out>.7:
            refined_path.append(img_path)
    return refined_path,confs




from train_utils_lit.inference_utils import get_loader,get_results
def gen_annotaions(checkpoint,iteration):
    ''' final funciton for getting annotations in pkl format'''
#     checkpoint='../input/phase1-training/Saved_weights/Mpii_pretrained/effnetb0epoch=68-valid_auc=0.4482-valid_acc=78.2685-train_acc=99.7902.ckpt'
    path='data/slp/train/train'
    ssl_img_paths,flipped_loader_ssl=get_loader(path=path,config=data_config,loader_type='slp',flip=True)
    _,notflipped_loader_ssl=get_loader(path=path,config=data_config,loader_type='slp',flip=False)
    result_ssl=get_results(model=lit_model,checkpoint=checkpoint,prediction_type='slp',
                     loadern=notflipped_loader_ssl,loaderf=flipped_loader_ssl)

    keypts=result_ssl[0]
    imgs=result_ssl[1]
    paths=ssl_img_paths

    predictions_ssl={}

    for pred,path in zip(keypts,paths):
        temp=pred.clone()

        temp[:,0]=((temp[:,0]/data_config['hm_size'][0])*120)
        temp[:,1]=((temp[:,1]/data_config['hm_size'][1])*160)
        predictions_ssl[path]=temp.numpy()
    save_obj(predictions_ssl,'annotations/pred{}'.format(iteration))









class Config:
    th=.5
    seed=42
    n_epoch=100
    #### schedular###
    lr = 1e-4
    max_lr = .9e-3
    pct_start = 0.3
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    ######
    betas=(0.9, 0.999)
    eps=1e-08
    weight_decay=0.01
    amsgrad=True
    steps_per_epoch=241
seed_everything(Config.seed)


data_config={
'input_size': (3, 388, 388),
 'interpolation': 'bicubic',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'crop_pct': 1.0,
 'hm_size':(104,104),
 'batch_size':1,
 'slp_train_path':'data/slp/train/train/',
 'slp_valid_path':'data/slp/valid/valid/',
 'mpii_images':'',
 'mpii_train_json':'',
 'mpii_valid_json': '',
 'unannotated_predictions':''

 }





# model = mxy

# lit_model = LitPose(
#     plConfig=Config,
#     data_config=data_config,
#     model=model,
#     phase=1
#     )
# # logger= WandbLogger(name='',project='Mpii training')  

# checkpoint_callback=ModelCheckpoint(monitor='valid_auc',
#                                    save_top_k=1,
#                                    save_last=True,
#                                    save_weights_only=False,
#                                    filename='{epoch:02d}-{valid_auc:.4f}-{valid_acc:.4f}-{train_loss:.4f}-{train_acc:.4f}',
#                                     verbose=False,
#                                     mode='max',
#                                     dirpath='./Saved_weights/Mpii_pretrained'
#                                    )




# trainer = Trainer(auto_lr_find=Config.lr,
#     max_epochs=Config.n_epoch,
#     gpus=[0],
#     callbacks=checkpoint_callback,
#     # logger=logger,

#     weights_summary='top',
#     amp_backend='native'
# )

# trainer.fit(lit_model)


for i in range(2):
    model = mxy
  
    if i==0:
            lit_model = LitPose(
            plConfig=Config,
            data_config=data_config,
            model=model,
            phase=1,
            preds=None,
            refined_path=None
            )
            logger= WandbLogger(name='iteration{},selected_img{}'.format(i,0),project='iterative training2') 
    else:
        weight=os.listdir('./preditions/iteration{}/'.format(i-1))
        checkpoint='./preditions/iteration{}/{}'.format((i-1),weight[0])
        gen_annotaions(checkpoint=checkpoint,iteration=i-1)
        predictions=load_obj('./preditions/pred{}.pkl'.format(i-1))
        refined_path,confs=get_inspected(predictions)
        
        lit_model = LitPose(
            plConfig=Config,
            data_config=data_config,
            model=model,
            phase=2,
            preds=predictions,
            refined_path=refined_path
            )
        logger= WandbLogger(name='iteration{},selected_img={}'.format(i,len(refined_path)),project='iterative training')  
        
    
    

    

    checkpoint_callback=ModelCheckpoint(monitor='valid_auc',
                                       save_top_k=1,
                                       save_last=False,
                                       save_weights_only=False,
                                       filename='effnetb0{epoch:02d}-{valid_auc:.4f}-{valid_acc:.4f}-{train_acc:.4f}',
                                        verbose=False,
                                        mode='max',
                                        dirpath='./preditions/iteration{}'.format(i)
                                       )




    trainer = Trainer(auto_lr_find=Config.lr,
        max_epochs=1,
        gpus=[0],
        callbacks=checkpoint_callback,
        # logger=logger,

        weights_summary='top',
        amp_backend='native'
    )
    checkpoint='pretrained_weights/effnet b0 epoch25-valid_auc0.6181-valid_acc84.2407-train_loss0.0002-train_acc88.2807.ckpt'
    lit_model.load_state_dict(torch.load(checkpoint)['state_dict'])

    trainer.fit(lit_model)
    logger.experiment.finish()