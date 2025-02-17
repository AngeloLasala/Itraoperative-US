"""
Unet base model to train conditional LDM
"""
import torch
from einops import einsum
import torch.nn as nn
from echocardiography.diffusion.models.blocks import get_time_embedding
from echocardiography.diffusion.models.blocks import DownBlock, MidBlock, UpBlockUnet
# from utils.config_utils import *

def get_config_value(config, key, default_value):
    """
    Get the value of a key from the config dictionary
    """
    return config[key] if key in config else default_value

class Unet(nn.Module):
    """
    Conditional Unet model comprising Down blocks, Midblocks and Uplocks.
    The conditional inputs can be class, text or image.
    -class condition: The class condition is a one-hot vector of size num_classes
    -text condition: The text condition is a sequence of tokens. 
                    it could be also speech, or image, beacuse it is conditioning with cross attention mechanism
    -image condition: The image condition is a mask image of size H x W     
    """
    
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        
        # Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        ######## Class, Mask and Text Conditioning Config #####
        self.class_cond = False
        self.class_relative = False
        self.text_cond = False
        self.image_cond = False
        self.keypoints_cond = False
        self.eco_parameters = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', default_value=None) 
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'class' in condition_types:
                # validate_class_config(self.condition_config)
                # print('sto dentro: class')
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'class_relative' in condition_types:
                # validate_class_config(self.condition_config)
                # print('sto dentro: class')
                self.class_relative = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'keypoints' in condition_types:
                # validate_keypoints_config(self.condition_config)
                # print('sto dentro: keypoints')
                self.keypoints_cond = True
                self.num_keypoints = self.condition_config['keypoints_condition_config']['num_keypoints']
            if 'eco_parameters' in condition_types:
                # validate_eco_parameters_config(self.condition_config)
                # print('sto dentro: eco_parameters')
                self.eco_parameters = True
                self.num_eco_parameters = self.condition_config['eco_parameters_condition_config']['num_eco_parameters']
            if 'text' in condition_types:
                # validate_text_config(self.condition_config)
                # print('sto dentro: text')
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config']['image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config']['image_condition_output_channels']

        if self.class_cond:
            # Rather than using a special null class we dont add the
            # class embedding information for unconditional generation
            self.class_emb = nn.Embedding(self.num_classes,
                                          self.t_emb_dim)

        if self.class_relative:
            # Rather than using a special null class we dont add
            # the class embedding information for unconditional generation
            self.class_emb = nn.Embedding(self.num_classes,
                                          self.t_emb_dim)
                            
        if self.keypoints_cond:
            # Rather than using a special null class we dont add
            # the keypoints embedding information for unconditional generation
            self.keypoints_emb = nn.Embedding(self.num_keypoints, self.t_emb_dim)

        if self.eco_parameters:
            # Rather than using a special null class we dont add
            # the eco_parameters embedding information for unconditional generation
            self.eco_parameters_emb = nn.Embedding(self.num_eco_parameters, self.t_emb_dim)
        
        if self.image_cond:
            # Map the mask image to a N channel image and
            # concat that with input across channel dimension
            self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
                                          out_channels=self.im_cond_output_ch,
                                          kernel_size=1,
                                          bias=False)
            self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch,
                                            self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        ## check if at least one condition is activate
        self.cond = self.text_cond or self.image_cond or self.class_cond or self.keypoints_cond or self.eco_parameters or self.class_relative
        ###################################
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])
        
        # Build the Downblocks
        for i in range(len(self.down_channels) - 1):
            # Cross Attention and Context Dim only needed if text condition is present
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels,
                                        cross_attn=self.text_cond,
                                        context_dim=self.text_embed_dim))
        
        self.mids = nn.ModuleList([])
        # Build the Midblocks
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      cross_attn=self.text_cond,
                                      context_dim=self.text_embed_dim))
                
        self.ups = nn.ModuleList([])
        # Build the Upblocks
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            norm_channels=self.norm_channels,
                            cross_attn=self.text_cond,
                            context_dim=self.text_embed_dim))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, cond_input=None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # print(f'Input) {x.shape}')
        if self.cond:
            assert cond_input is not None, "Model initialized with conditioning so cond_input cannot be None"
        if self.image_cond:
            # print('sto dentro al image_cond')
            ######## Mask Conditioning ########
            # validate_image_conditional_input(cond_input, x)
            im_cond = cond_input['image']
            im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            assert im_cond.shape[-2:] == x.shape[-2:]
            x = torch.cat([x, im_cond], dim=1)
            # B x (C+N) x H x W
            out = self.conv_in_concat(x)
            # print(f'conv_in_concat) {out.shape}')
            #####################################
        else:
            # B x C x H x W
            out = self.conv_in(x)
        # B x C1 x H x W
        # print(f'first conv) {out.shape}')
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        ######## Class Conditioning ########
        if self.class_cond:
            # validate_class_conditional_input(cond_input, x, self.num_classes)
            # print(self.class_emb.weight.shape)
            # print(cond_input['class'].shape)
            class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
            # print(class_embed.shape)
            # print(t_emb.shape)
            t_emb += class_embed

        ######## Class Relative Conditioning ########
        if self.class_relative:
            # validate_class_conditional_input(cond_input, x, self.num_classes)
            # print(self.class_emb.weight.shape)
            # print(cond_input['class'].shape)
            class_embed = einsum(cond_input['class_relative'].float(), self.class_emb.weight, 'b n, n d -> b d')
            # print(class_embed.shape)
            # print(t_emb.shape)
            t_emb += class_embed
        ####################################

        ######## Eco Parameters Conditioning ########
        if self.eco_parameters:
            # validate_eco_parameters_conditional_input(cond_input, x, self.num_eco_parameters)
            # print(self.eco_parameters_emb.weight.shape)
            # print(cond_input['eco_parameters'].shape)
            eco_parameters_embed = einsum(cond_input['eco_parameters'].float(), self.eco_parameters_emb.weight, 'b n, n d -> b d')
            # print(eco_parameters_embed.shape)
            # print(t_emb.shape)
            t_emb += eco_parameters_embed

        ######## Keypoints Conditioning ########
        if self.keypoints_cond:
            # validate_keypoints_conditional_input(cond_input, x, self.num_keypoints)
            # print(self.keypoints_emb.weight.shape)
            # print(cond_input['keypoints'].shape)
            keypoints_embed = einsum(cond_input['keypoints'].float(), self.keypoints_emb.weight, 'b n, n d -> b d')
            # print(keypoints_embed.shape)
            # print(t_emb.shape)
            t_emb += keypoints_embed
        ####################################
            
        context_hidden_states = None
        if self.text_cond:
            assert 'text' in cond_input, \
                "Model initialized with text conditioning but cond_input has no text information"
            context_hidden_states = cond_input['text']        
        down_outs = []
        
        # print('DOWN BLOCKS')
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            # print(f'idx: {idx} - out: {out.shape} - t_emb: {t_emb.shape} - context_hidden_states: {context_hidden_states.shape}')
            out = down(out, t_emb, context_hidden_states)
            

        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        # print()
        
        # print('MID BLOCKS')
        for idx, mid in enumerate(self.mids):
            out = mid(out, t_emb, context_hidden_states)
            # print(f'idx: {idx} - out: {out.shape} - t_emb: {t_emb.shape} - context_hidden_states: {context_hidden_states.shape}')
        # out B x C3 x H/4 x W/4
        # print()
        
        # print('UP BLOCKS')
        for idx,up in enumerate(self.ups):
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out

if __name__ == '__main__':
    model_config = {
        'down_channels': [ 128, 256, 256, 256],
        'mid_channels': [ 256, 256],
        'down_sample': [ False, True, False],
        'attn_down' : [True, True, True],
        'time_emb_dim': 256,
        'norm_channels' : 32,
        'num_heads' : 16,
        'conv_out_channels' : 128,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
        'condition_config': {
            'condition_types': ['class_relative'],
            'class_condition_config': {
                'num_classes': 4
            },
            'text_condition_config': {
                'image_condition_input_channels': 1,  
                'text_embed_dim' : 768,
                'text_embed_prob' : 0.1,
                
            },
            'image_condition_config': {
                'image_condition_input_channels': 6,
                'image_condition_output_channels': 3,
            },
            'keypoints_condition_config': {
                'num_keypoints': 12
            },
            'eco_parameters_condition_config': {
                'num_eco_parameters': 2
            }
        }
    }
    
    model = Unet(3, model_config)
    x = torch.randn(5, 3, 30, 40)
    t = torch.randint(0, 100, (5,))
    text_cond = torch.randn(5, 8, 8, 768)
    mask_cond = torch.randn(5, 6, 30, 40)
    class_cond = torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0], [0,0,0,1]])
    class_relative_cond = torch.tensor([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0], [0,0,0,1]])
    keypoints_cond = torch.randn(5, 12)
    eco_parameters_cond = torch.randn(5, 2)
    out = model(x, t, {'class_relative': class_relative_cond})
    print(out.shape)