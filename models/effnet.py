import os
import torch
import torch.nn as nn
import timm
import logging
logger = logging.getLogger(__name__)
class Modelx(nn.Module):
    
    def __init__(self, model_name, pretrained=True):
        
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()    


    def forward(self, x):
        x = self.model(x)
        
        return x

class Modely(nn.Module):
    def __init__(self, num_joints = 14):
        super(Modely, self).__init__()
        self.inplanes = 1280
        self.BN_MOMENTUM = 0.1
        self.deconv_with_bias = False
        self.dconv_dict = self.get_dconv_config()
        self.num_joints = num_joints
#         NUM_DECONV_LAYERS = 3
#         NUM_DECONV_KERNELS = [4, 4, 4]
#         NUM_DECONV_FILTERS = [256, 256, 256]
        
        FINAL_CONV_KERNEL = 1


        self.deconv_layers = self._make_deconv_layer()
        
        self.final_layer = nn.Conv2d(
            in_channels = self.dconv_dict['out_channel'][-1],
            out_channels = self.num_joints,
            kernel_size= FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if FINAL_CONV_KERNEL == 3 else 0
        )

    def get_dconv_config(self):
        return {
            'n_dconv' : 3,
            'kernels' : [4, 4, 4],
            'strides' : [2, 2, 2],
            'padding' : [1, 1, 1],
            'out_padding' : [0,0,0],
            'out_channel' : [256, 256, 256]
        }
#     def _get_deconv_cfg(self, deconv_kernel, index):
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0

#         return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self):
#         assert num_layers == len(num_filters), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'
#         assert num_layers == len(num_kernels), \
#             'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(self.dconv_dict['n_dconv']):
            kernel = self.dconv_dict['kernels'][i]
            padding = self.dconv_dict['padding'][i]
            output_padding = self.dconv_dict['out_padding'][i]
            stride = self.dconv_dict['strides'][i]

                
            planes = self.dconv_dict['out_channel'][i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x


class Modelxy(nn.Module):
    def __init__(self,model_name):
        super(Modelxy, self).__init__()

        self.model_name=model_name
        self.modelx=Modelx(model_name=self.model_name)
        self.modely=Modely()

    def forward(self,x):
        x=self.modelx(x)
        x=self.modely(x)


        return x
    
    def save(self,epoch):
        self.eval()
        torch.save({
            'epoch': epoch,
            'state_dict': self.state_dict(),
            # 'optimizer_state_dict': optim.state_dict(),
            'loss': 0,
            }, 'modelxy{}.pth'.format(epoch))

    # def load(self,optim,path):
    #     checkpoint = torch.load(path)
    #     self.load_state_dict(checkpoint['model_state_dict'])
    #     optim.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.epoch = checkpoint['epoch']
    #     self.loss = checkpoint['loss']

    def load(self,cfg, path):
        if os.path.isfile(path):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.modely.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.modely.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.modely.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(path))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(path)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')
        cfg.TRAIN.BEGIN_EPOCH = checkpoint['epoch']


def get_pose_net(is_train = True):
    model_name = 'efficientnet_b0'

    model = Modelxy(model_name)
    # print(torch.load(cfg.MODEL.PRETRAINED).keys())
    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #     model.load(cfg, cfg.MODEL.PRETRAINED)

    return model
print('model done1')
mxy=get_pose_net()
print('model done2')
print(mxy(torch.zeros(1,3,440,440)).shape)