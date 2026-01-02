"""
Unet base model to train LDM
"""
import torch
import torch.nn as nn
from intraoperative_us.diffusion.models.blocks import get_time_embedding
from intraoperative_us.diffusion.models.blocks import DownBlock, MidBlock, UpBlockUnet
import torch
import torch.nn as nn
from diffusers import UNet2DModel
from thop import profile

class Unet(nn.Module):
    """
    Unet model comprising
    Down blocks, Midblocks and Uplocks
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
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                                    self.t_emb_dim, up_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_up_layers,
                                        norm_channels=self.norm_channels))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
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
        'down_sample': [ False, False, False ],
        'attn_down' : [True, True, True],
        'time_emb_dim': 256,
        'norm_channels' : 32,
        'num_heads' : 16,
        'conv_out_channels' : 128,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
    }
    model = Unet(3, model_config)
    x = torch.randn(16, 3, 30, 40)
    t = torch.randint(0, 100, (16,))
    out = model(x, t)
    print(out.shape)
    # print(out)


    # Test with diffusers UNet2DModel
    def get_number_parameter(model):
        """
        Count the total number of trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print("=" * 60)
        print("Model Parameters")
        print("=" * 60)
        print(f"Total Parameters:         {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"Trainable Parameters:     {trainable_params:,} ({trainable_params/1e6:.2f} M)")
        print(f"Non-trainable Parameters: {non_trainable_params:,}")
        print(f"Model Size (MB):          {total_params * 4 / (1024**2):.2f}")
        print("=" * 60)
        
        return total_params, trainable_params, non_trainable_params


    def compute_gflops(model, input_shape=(1, 4, 32, 32), is_diffusers_model=True):
        """
        Compute GFLOPs using thop library.
        
        Args:
            model: The model to analyze
            input_shape: Tuple of (batch_size, channels, height, width)
            is_diffusers_model: Whether the model is from diffusers library
        
        Returns:
            gflops (float): Estimated GFLOPs
            params (int): Number of parameters
        """
        device = next(model.parameters()).device
        
        # Prepare inputs
        x = torch.randn(input_shape).to(device)
        t = torch.randint(0, 1000, (input_shape[0],)).to(device)
        
        if is_diffusers_model:
            # For diffusers models, we need to wrap the forward call
            class DiffusersWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x, t):
                    return self.model(x, t).sample
            
            wrapped_model = DiffusersWrapper(model).to(device)
            macs, params = profile(wrapped_model, inputs=(x, t), verbose=False)
        else:
            # For custom models
            macs, params = profile(model, inputs=(x, t), verbose=False)
        
        # Convert MACs to GFLOPs (1 MAC ≈ 2 FLOPs)
        gflops = macs * 2 / 1e9
        
        return gflops, params


    def compute_model_complexity(model, input_shape=(1, 4, 32, 32), is_diffusers_model=True):
        """
        Compute and print complete model complexity metrics.
        
        Args:
            model: The model to analyze
            input_shape: Tuple of (batch_size, channels, height, width)
            is_diffusers_model: Whether the model is from diffusers library
        """
        print("\n" + "=" * 60)
        print("MODEL COMPLEXITY ANALYSIS")
        print("=" * 60)
        
        # Count parameters
        total_params, trainable_params, non_trainable_params = get_number_parameter(model)
        
        # Compute GFLOPs
        print("\nComputing GFLOPs...")
        try:
            gflops, params_from_thop = compute_gflops(model, input_shape, is_diffusers_model)
            print("=" * 60)
            print("Computational Complexity")
            print("=" * 60)
            print(f"Input Shape:              {input_shape}")
            print(f"GFLOPs per forward pass:  {gflops:.2f}")
            print(f"Parameters (from thop):   {params_from_thop:,} ({params_from_thop/1e6:.2f} M)")
            print("=" * 60)
        except Exception as e:
            print(f"Could not compute GFLOPs: {e}")
            print("=" * 60)


    model = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        block_out_channels=[128, 256, 256, 256]
    )
    
    # Test forward pass
    x = torch.randn(1, 4, 32, 32)
    t = torch.randint(0, 100, (1,))
    out = model(x, t).sample
    print(f"\nOutput shape: {out.shape}\n")
    
    # Compute model complexity
    compute_model_complexity(model, input_shape=(1, 4, 32, 32), is_diffusers_model=True)