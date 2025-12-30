import torch
import json
from pathlib import Path
import time
import numpy as np

# Carica il piano
plans_file = "nnUNetPlans.json"
plans_path = Path(plans_file)

if not plans_path.exists():
    raise FileNotFoundError(f"File non trovato: {plans_file}")

with open(plans_path, 'r') as f:
    plans_data = json.load(f)

# Accedi alla configurazione 2D
config = plans_data["configurations"]["2d"]
arch_config = config["architecture"]["arch_kwargs"]

# Estrai i parametri dell'architettura
n_stages = arch_config["n_stages"]
features_per_stage = arch_config["features_per_stage"]
kernel_sizes = arch_config["kernel_sizes"]
strides = arch_config["strides"]
n_conv_per_stage = arch_config["n_conv_per_stage"]
n_conv_per_stage_decoder = arch_config["n_conv_per_stage_decoder"]

# Parametri input/output
input_channels = 1
num_classes = 2

print(f"Configurazione rilevata dal piano:")
print(f"  - Dataset: {plans_data['dataset_name']}")
print(f"  - Tipo: 2D")
print(f"  - N stages: {n_stages}")
print(f"  - Features per stage: {features_per_stage}")
print(f"  - Patch size: {config['patch_size']}")
print(f"  - Batch size: {config['batch_size']}")

# Importa dinamicamente le classi necessarie
from dynamic_network_architectures.architectures.unet import PlainConvUNet

# Crea il modello
model = PlainConvUNet(
    input_channels=input_channels,
    n_stages=n_stages,
    features_per_stage=features_per_stage,
    conv_op=torch.nn.Conv2d,
    kernel_sizes=kernel_sizes,
    strides=strides,
    n_conv_per_stage=n_conv_per_stage,
    num_classes=num_classes,
    n_conv_per_stage_decoder=n_conv_per_stage_decoder,
    conv_bias=arch_config["conv_bias"],
    norm_op=torch.nn.InstanceNorm2d,
    norm_op_kwargs=arch_config["norm_op_kwargs"],
    dropout_op=None,
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs=arch_config["nonlin_kwargs"]
)

# Calcola parametri
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Funzione per calcolare GFLOPs
def count_conv2d_flops(module, input, output):
    """Calcola FLOPs per layer Conv2d"""
    input = input[0]
    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]
    
    kernel_height, kernel_width = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups
    
    flops = batch_size * output_height * output_width * \
            (in_channels / groups) * out_channels * \
            kernel_height * kernel_width * 2
    
    module.__flops__ += int(flops)

def count_normalization_flops(module, input, output):
    """Calcola FLOPs per normalization layers"""
    input = input[0]
    flops = input.numel() * 3
    module.__flops__ += int(flops)

def count_activation_flops(module, input, output):
    """Calcola FLOPs per activation functions"""
    flops = output.numel()
    module.__flops__ += int(flops)

def calculate_flops(model, input_shape, device):
    """Calcola i GFLOPs totali del modello"""
    model_copy = type(model)(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=torch.nn.Conv2d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=arch_config["conv_bias"],
        norm_op=torch.nn.InstanceNorm2d,
        norm_op_kwargs=arch_config["norm_op_kwargs"],
        dropout_op=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs=arch_config["nonlin_kwargs"]
    ).to(device)
    
    model_copy.eval()
    
    # Inizializza contatori
    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        
        m.__flops__ = 0
        
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(count_conv2d_flops)
        elif isinstance(m, (torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d)):
            m.register_forward_hook(count_normalization_flops)
        elif isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU)):
            m.register_forward_hook(count_activation_flops)
    
    model_copy.apply(add_hooks)
    
    # Forward pass
    input_tensor = torch.randn(input_shape).to(device)
    with torch.no_grad():
        model_copy(input_tensor)
    
    # Somma tutti i FLOPs
    total_flops = 0
    for m in model_copy.modules():
        if hasattr(m, '__flops__'):
            total_flops += m.__flops__
    
    del model_copy
    return total_flops

def benchmark_inference(model, input_shape, device, num_iterations=100, warmup=10):
    """Esegue benchmark del tempo di inferenza"""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up
    print(f"  Warming up on {device}...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Sincronizza
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Misura tempo di inferenza
    print(f"  Measuring inference time on {device}...")
    inference_times = []
    
    for _ in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000)  # ms
    
    # Calcola statistiche
    results = {
        'mean': np.mean(inference_times),
        'std': np.std(inference_times),
        'min': np.min(inference_times),
        'max': np.max(inference_times),
        'median': np.median(inference_times),
        'fps': 1000 / np.mean(inference_times),
        'output_shape': list(output.shape)
    }
    
    return results

# Calcola GFLOPs (indipendente dal device)
patch_size = config['patch_size']
input_shape = (1, input_channels, patch_size[0], patch_size[1])

print("\nCalcolando GFLOPs...")
total_flops = calculate_flops(model, input_shape, torch.device('cpu'))
gflops = total_flops / 1e9

# Stampa info generali
print(f"\n{'='*70}")
print(f"Statistiche del modello nnUNetv2 - 2D")
print(f"{'='*70}")
print(f"Dataset:             {plans_data['dataset_name']}")
print(f"Input shape:         {list(input_shape)}")
print(f"-" * 70)
print(f"Input channels:      {input_channels}")
print(f"Output classes:      {num_classes}")
print(f"Num stages:          {n_stages}")
print(f"Features per stage:  {features_per_stage}")
print(f"-" * 70)
print(f"Parametri totali:    {total_params / 1e6:.2f} M ({total_params:,})")
print(f"Parametri trainable: {trainable_params / 1e6:.2f} M ({trainable_params:,})")
print(f"GFLOPs:              {gflops:.2f}")
print(f"{'='*70}\n")

# Benchmark su CPU
print("=" * 70)
print("BENCHMARK CPU")
print("=" * 70)
cpu_results = benchmark_inference(model, input_shape, torch.device('cpu'), 
                                  num_iterations=100, warmup=10)
print(f"Output shape:        {cpu_results['output_shape']}")
print(f"-" * 70)
print(f"Tempo di inferenza (100 iterations):")
print(f"  Mean:              {cpu_results['mean']:.2f} ms")
print(f"  Median:            {cpu_results['median']:.2f} ms")
print(f"  Std:               {cpu_results['std']:.2f} ms")
print(f"  Min:               {cpu_results['min']:.2f} ms")
print(f"  Max:               {cpu_results['max']:.2f} ms")
print(f"  FPS:               {cpu_results['fps']:.2f}")
print(f"{'='*70}\n")

# Benchmark su GPU (se disponibile)
if torch.cuda.is_available():
    print("=" * 70)
    print(f"BENCHMARK GPU ({torch.cuda.get_device_name(0)})")
    print("=" * 70)
    gpu_results = benchmark_inference(model, input_shape, torch.device('cuda'), 
                                      num_iterations=100, warmup=10)
    print(f"Output shape:        {gpu_results['output_shape']}")
    print(f"-" * 70)
    print(f"Tempo di inferenza (100 iterations):")
    print(f"  Mean:              {gpu_results['mean']:.2f} ms")
    print(f"  Median:            {gpu_results['median']:.2f} ms")
    print(f"  Std:               {gpu_results['std']:.2f} ms")
    print(f"  Min:               {gpu_results['min']:.2f} ms")
    print(f"  Max:               {gpu_results['max']:.2f} ms")
    print(f"  FPS:               {gpu_results['fps']:.2f}")
    print(f"-" * 70)
    print(f"Speedup GPU vs CPU: {cpu_results['mean'] / gpu_results['mean']:.2f}x")
    print(f"{'='*70}\n")
else:
    print("GPU non disponibile. Benchmark solo su CPU.\n")

# Salva risultati su file
results_summary = {
    'dataset': plans_data['dataset_name'],
    'input_shape': list(input_shape),
    'total_params': total_params,
    'trainable_params': trainable_params,
    'gflops': gflops,
    'cpu': cpu_results
}

if torch.cuda.is_available():
    results_summary['gpu'] = gpu_results
    results_summary['speedup'] = cpu_results['mean'] / gpu_results['mean']

output_file = 'model_complexity_results.json'
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=4)

print(f"Risultati salvati in: {output_file}")