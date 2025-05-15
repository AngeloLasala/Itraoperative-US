"""
Unet base model to train conditional LDM
"""
import os
import torch
from einops import einsum
import torch.nn as nn
from intraoperative_us.diffusion.models.blocks import get_time_embedding
from intraoperative_us.diffusion.models.blocks import DownBlock, MidBlock, UpBlockUnet
from intraoperative_us.diffusion.utils.utils import get_number_parameter
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from intraoperative_us.diffusion.utils.utils import get_number_parameter



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
        self.text_cond = False
        self.image_cond = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', default_value=None) 
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'text' in condition_types:
                # validate_text_config(self.condition_config)
                # print('sto dentro: text')
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config']['image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config']['image_condition_output_channels']


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
        self.cond = self.text_cond or self.image_cond
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

class MaskConcatenationModel(nn.Module):
    """
    Convolutional model to process mask for the concatenation with the latent space
    """
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_hint_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, self.out_channels, 3, padding=1),
        )

    def forward(self, hint: torch.Tensor):
        # image
        hint = self.input_hint_block(hint)
        return hint

class MaskEmbedding(nn.Module):
    """
    Custom model to get the embedding of the mask image.
    """
    def __init__(self, in_channels: int, img_embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.img_embed_dim = img_embed_dim
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, 4, 3, padding=1),  # Output shape: (B, 4, H/8, W/8)
        )
        self.image_proj = nn.Linear(4, img_embed_dim)  # Maps (B, 4) → (B, img_embed_dim)
        self.image_norm = nn.LayerNorm(img_embed_dim)

    def forward(self, hint: torch.Tensor):
        # image
        hint = self.input_hint_block(hint)
        hint = hint.mean(dim=[2, 3])       # Global Average Pooling → (B, 4)
        hint = self.image_proj(hint)
        hint = self.image_norm(hint)
        return hint

# create a class to load 2dUnet model from diffuser
class UNet2DConditionModelCostum(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.unet_input_ch = model_config['z_channels']

        ## Define condition
        self.image_cond = False
        self.condition_config = get_config_value(model_config, 'condition_config', default_value=None) 
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'image' in condition_types:
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config['image_condition_config']['image_condition_input_channels']
                self.im_cond_output_ch = self.condition_config['image_condition_config']['image_condition_output_channels']
                self.unet_input_ch = model_config['z_channels'] + self.im_cond_output_ch
                self.mask_model = MaskConcatenationModel(in_channels=self.im_cond_input_ch, out_channels=self.im_cond_output_ch)

        ## UNet model for diffusion process
        if model_config['initialization'] == 'random':
            print('Training UNet with random initialization')
            self.model = UNet2DConditionModel(
                sample_size=model_config['sample_size'],
                in_channels=self.unet_input_ch,
                out_channels=model_config['z_channels'],
                block_out_channels=model_config['down_channels'],
                cross_attention_dim=model_config['cross_attention_dim'],
            )

        elif model_config['initialization'] == 'SD1.5' or model_config['initialization'] == 'lora':
            print('Training UNet with pretrained Hugginface model SDv1.5')
            self.model = UNet2DConditionModel.from_pretrained(os.path.join(model_config['unet_path'], model_config['unet']),
                                                          sample_size=model_config['sample_size'],
                                                          in_channels=self.unet_input_ch,
                                                          out_channels=model_config['z_channels'],
                                                          block_out_channels=model_config['down_channels'],
                                                          cross_attention_dim=model_config['cross_attention_dim'],
                                                          low_cpu_mem_usage=False,
                                                          use_safetensors=True,
                                                          ignore_mismatched_sizes=True)
    
    def forward(self, x, t, context_hidden_states, cond_input):

        ######## Mask Conditioning ########
        if self.image_cond:
            # validate_image_conditional_input(cond_input, x)
            im_cond = cond_input['image']
            im_cond = self.mask_model(im_cond)
            assert im_cond.shape[-2:] == x.shape[-2:]
            x = torch.cat([x, im_cond], dim=1)

        out = self.model(x, t, context_hidden_states, return_dict=False)[0]
        
        return  out


    



if __name__ == '__main__':
    model_config = {
        'unet': 'unet_cond/UNet2DConditionModel_SD1.5_default',     ## path of hugginface model
        ## local : "/home/angelo/Documenti/Itraoperative-US/intraoperative_us/diffusion/models"
        ## cineca: "/leonardo_work/IscrC_Med-LMGM/Angelo/Itraoperative-US/intraoperative_us/diffusion/models"
        'unet_path': "/home/angelo/Documenti/Itraoperative-US/intraoperative_us/diffusion/models",
        # in_channels: 4,  i.e. z_channels
        # out_channels: 4, i.e. z_channels
        'initialization': 'random',
        'sample_size': 32,                    ## shape of the input of autoencoder
        'z_channels': 4,                           ## z_channels of the autoencoder  
        'down_channels': [ 128, 256, 256, 256],
        'mid_channels': [ 256, 256],
        'down_sample': [ True, True, True],
        'attn_down' : [True, True, True],
        'time_emb_dim': 256,
        'norm_channels' : 32,
        'num_heads' : 16,
        'conv_out_channels' : 128,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
        'cross_attention_dim' : 768
        'condition_config': {
            'condition_types': ['image'],
            'text_condition_config': {
                'image_condition_input_channels': 1,  
                'text_embed_dim' : 768,
                'text_embed_prob' : 0.1,
                
            },
            'image_condition_config': {
                'image_condition_input_channels': 1,
                'image_condition_output_channels': 4,
            }
        }
    }
    
    model = Unet(3, model_config)
    x = torch.randn(5, 3, 32, 32)
    t = torch.randint(0, 100, (5,))
    text_cond = torch.randn(5, 8, 8, 768)
    mask_cond = torch.randn(5, 1, 32, 32)
    keypoints_cond = torch.randn(5, 12)
    eco_parameters_cond = torch.randn(5, 2)
    out = model(x, t, {'image': mask_cond})
    get_number_parameter(model)
    print(out.shape)
    get_number_parameter(model)

    mask = torch.randn(5, 1, 256, 256)
    mask_model = MaskConcatenationModel(in_channels=1, out_channels=4)
    out = mask_model(mask)
    print(out.shape)

    mask_embed = MaskEmbedding(in_channels=1, img_embed_dim=768)
    out = mask_embed(mask)
    print(out.shape)
    print()

    ## costum mode
    unet = UNet2DConditionModelCostum(model_config)
    latent = torch.randn(5, 4, 32, 32)
    mask = torch.randn(5, 1, 256, 256)
    context_hidden_states = torch.randn(5, 77, 768)

    out = unet(latent, 1, context_hidden_states, {'image': mask})
    get_number_parameter(unet)
    print(out.shape)

    ## load 2D conditional unet model from diffuser
    # unet = UNet2DConditionModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True)
    # unet = UNet2DConditionModel.from_pretrained("unet_cond/UNet2DConditionModel_SD1.5_default",
    #                                             addition_embed_type='image',
    #                                             encoder_hid_dim=256,
    #                                             low_cpu_mem_usage=False,
    #                                             use_safetensors=True)

    # tokenizer = CLIPTokenizer.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="text_encoder", use_safetensors=True)
    # tokenizer.save_pretrained("tokenizer/CLIPTokenizer")
    # text_encoder.save_pretrained("text_encoder/CLIPTextModel")
    # get_number_parameter(unet)
    # # unet.save_pretrained("unet_cond/UNet2DConditionModel_SD1.5_default")

    
    # for k in unet.config.keys():
    #     print(k, unet.config[k])
    