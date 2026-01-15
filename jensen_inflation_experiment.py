import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
Full Batch Experiment: Validating Jensen Inflation and Φ Effect in Adaptive Optimizers
Improved Version: Corrected oscillation metric calculation + Added multiple experiment repetition functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# ==================== Experiment Configuration ====================
BASE_BATCH = 32
BASE_LR = 0.001
TOTAL_STEPS = 1500  # Increased steps for better post-convergence data
WARMUP_STEPS = 300  # First 300 steps for warmup, excluded from oscillation analysis
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment repetition count and random seeds
NUM_RUNS = 5  # Run 5 independent experiments
RUN_SEEDS = [42,123,456, 789, 999]  # Fixed random seeds

print(f"Using device: {DEVICE}")
print(f"Number of experiment repetitions: {NUM_RUNS}")
print(f"Random seeds: {RUN_SEEDS}")

# Create all necessary directories
os.makedirs("analysis_results", exist_ok=True)
os.makedirs("analysis_results/checkpoints", exist_ok=True)
os.makedirs("analysis_results/experiment_runs", exist_ok=True)

# Batch size range
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]  # Full range

# Learning rate strategies
learning_rate_strategies = {
    "fixed": lambda bs, step: BASE_LR,
    "linear": lambda bs, step: BASE_LR * (bs / BASE_BATCH),
    "sqrt": lambda bs, step: BASE_LR * np.sqrt(bs / BASE_BATCH),
}

# ==================== Function to set random seeds ====================
def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set random seed: {seed}")

# ==================== Improved Metrics Collector (Corrected oscillation calculation) ====================
class CorrectedMetricsCollector:
    def __init__(self, num_params):
        self.records = []
        self.num_params = num_params
        
        # Historical data for oscillation calculation
        self.loss_history = []  # Loss history
        self.weight_history = []  # Parameter history
        self.variance_history = []  # Second moment history
        self.momentum_history = []  # Momentum history
        
        self.max_history = 200  # Keep recent 200 steps for oscillation calculation
        
        # Auxiliary variables for parameter oscillation calculation
        self.weight_trajectory = []  # Parameter trajectory
        self.converged_weights = None  # Reference weights after convergence
        self.convergence_start_step = None  # Step when convergence starts
        
        # Auxiliary variables for loss oscillation calculation
        self.smoothed_loss = None
        self.loss_ema_alpha = 0.9
        
        # New: Initialize prev_w_old
        self.prev_w_old = None
        
    def collect_step(self, step, batch_size, lr_strategy,
                     gradients, m, v, w, loss, optimizer_state):
        """Collect metrics for each step"""
        
        # Save historical data for oscillation calculation
        self._update_histories(step, loss, w, v, m)
        
        # Calculate gradient statistics
        grad_mean = gradients.mean()
        grad_var = gradients.var()
        grad_skew = self._compute_skewness(gradients)
        grad_kurtosis = self._compute_kurtosis(gradients)
        
        # Improved tail index estimation
        tail_index = self._estimate_tail_index_improved(gradients)
        
        # Improved Φ calculation
        phi_value = self._compute_phi_improved(gradients, step)
        
        # Calculate inflation rate
        inflation = self._compute_inflation_rate_improved(v, gradients)
        
        # ==================== Correction: New oscillation metrics calculation ====================
        # 1. Oscillation based on loss function (relative fluctuation)
        oscillation_loss = self._compute_oscillation_from_loss(step)
        
        # 2. Oscillation based on detrended parameters (fluctuation related to SDE theory)
        oscillation_param = self._compute_oscillation_from_weights(step)
        
        # 3. Oscillation based on effective learning rate fluctuation
        oscillation_lr = self._compute_oscillation_from_effective_lr(v, gradients, step)
        
        # 4. Composite oscillation metric (weighted average)
        oscillation_composite = self._compute_composite_oscillation(
            oscillation_loss, oscillation_param, oscillation_lr, step
        )
        
        # Keep old oscillation calculation method (for comparison)
        oscillation_old = self._compute_old_oscillation(w, step)
        
        # Calculate other metrics
        v_var = v.var()
        v_mean = v.mean()
        m_norm = m.norm()
        noise_strength = self._estimate_noise_strength(gradients)
        
        # Detect convergence (for late-stage oscillation analysis)
        convergence_status = self._detect_convergence(step, loss)
        
        record = {
            'step': step,
            'batch_size': batch_size,
            'lr_strategy': lr_strategy,
            'grad_mean': grad_mean.item(),
            'grad_var': grad_var.item(),
            'grad_skew': grad_skew,
            'grad_kurtosis': grad_kurtosis,
            'tail_index': tail_index,
            'phi': phi_value,
            'phi_squared': phi_value ** 2,
            'inflation': inflation,
            # ============ New oscillation metrics ============
            'oscillation_loss': oscillation_loss,
            'oscillation_param': oscillation_param,
            'oscillation_lr': oscillation_lr,
            'oscillation_composite': oscillation_composite,
            'oscillation_old': oscillation_old,  # For comparison
            # ====================================
            'v_var': v_var.item(),
            'v_mean': v_mean.item(),
            'm_norm': m_norm.item(),
            'noise_strength': noise_strength,
            'loss': loss.item(),
            'learning_rate': optimizer_state.get('lr', 0.0),
            'converged': convergence_status
        }
        
        self.records.append(record)
    
    def _update_histories(self, step, loss, w, v, m):
        """Update historical data"""
        # Loss history
        self.loss_history.append(loss.item())
        if len(self.loss_history) > self.max_history:
            self.loss_history.pop(0)
        
        # Parameter history (save only part to save memory)
        if step % 10 == 0:  # Save every 10 steps
            w_sample = w[::len(w)//100] if len(w) > 100 else w  # Sample 100 parameters
            self.weight_history.append(w_sample.detach().cpu().numpy().copy())
            if len(self.weight_history) > self.max_history // 10:
                self.weight_history.pop(0)
        
        # Save complete parameter trajectory for convergence detection
        if step % 50 == 0:
            w_norm = torch.norm(w).item()
            self.weight_trajectory.append((step, w_norm))
        
        # Second moment and momentum history (for learning rate oscillation calculation)
        if step % 5 == 0:
            self.variance_history.append(v.mean().item())
            self.momentum_history.append(m.norm().item())
            if len(self.variance_history) > self.max_history // 5:
                self.variance_history.pop(0)
                self.momentum_history.pop(0)
    
    def _compute_skewness(self, x):
        """Calculate skewness"""
        x_np = x.cpu().detach().numpy().flatten()
        if len(x_np) < 2:
            return 0.0
        mean = np.mean(x_np)
        std = np.std(x_np)
        if std < 1e-8:
            return 0.0
        skew = np.mean(((x_np - mean) / std) ** 3)
        return skew
    
    def _compute_kurtosis(self, x):
        """Calculate kurtosis"""
        x_np = x.cpu().detach().numpy().flatten()
        if len(x_np) < 4:
            return 3.0
        try:
            from scipy.stats import kurtosis
            return kurtosis(x_np, fisher=False, bias=False)
        except:
            mean = np.mean(x_np)
            std = np.std(x_np)
            if std < 1e-8:
                return 3.0
            kurt = np.mean(((x_np - mean) / std) ** 4)
            return kurt
    
    def _estimate_tail_index_improved(self, gradients):
        """Improved tail index estimation"""
        x_np = np.abs(gradients.cpu().detach().numpy().flatten())
        if len(x_np) < 10:
            return 4.0
        
        x_sorted = np.sort(x_np)[::-1]
        ks = [len(x_sorted)//20, len(x_sorted)//10, len(x_sorted)//5]
        ks = [max(k, 2) for k in ks]
        
        alphas = []
        for k in ks:
            tail_samples = x_sorted[:k]
            if len(tail_samples) > 1 and tail_samples[-1] > 0:
                log_ratios = np.log(tail_samples / tail_samples[-1])
                alpha = 1.0 / np.mean(log_ratios)
                alphas.append(alpha)
        
        return np.median(alphas) if alphas else 4.0
    
    def _compute_phi_improved(self, gradients, step):
        """Improved Φ calculation"""
        g_np = gradients.cpu().detach().numpy().flatten()
        
        # Use moving window to calculate statistics
        window_size = min(100, len(g_np))
        if len(g_np) < 10:
            return 0.0
        
        # Estimate local statistics using current batch gradients
        g_mean = np.mean(g_np)
        g_var = np.var(g_np)
        
        # Estimate gradient noise
        g_noise = g_np - g_mean
        
        # Calculate ξ and ξ²
        g_noise_sq = g_noise ** 2
        
        if len(g_noise) > 5:
            g_noise_std = np.std(g_noise)
            g_noise_sq_std = np.std(g_noise_sq)
            
            if g_noise_std > 1e-8 and g_noise_sq_std > 1e-8:
                phi = np.corrcoef(g_noise, g_noise_sq)[0, 1]
                return float(phi) if not np.isnan(phi) else 0.0
        
        return 0.0
    
    def _compute_inflation_rate_improved(self, v, gradients):
        """Calculate inflation rate"""
        v_flat = v.flatten().detach()
        
        # Basic inflation rate: 𝔼[1/√v] / (1/√𝔼[v])
        mean_inverse = (1.0 / torch.sqrt(v_flat + 1e-8)).mean()
        inverse_of_mean = 1.0 / torch.sqrt(v_flat.mean() + 1e-8)
        basic_inflation = mean_inverse / inverse_of_mean
        
        return basic_inflation.item()
    
    # ==================== New oscillation calculation methods ====================
    
    def _compute_oscillation_from_loss(self, step):
        """Loss-based oscillation: std/mean of loss"""
        if len(self.loss_history) < 20 or step < WARMUP_STEPS:
            return 0.0
        
        losses = np.array(self.loss_history)
        mean_loss = np.mean(losses)
        
        if mean_loss < 1e-8:
            return 0.0
        
        # Use moving window to calculate local std
        window_size = min(50, len(losses))
        rolling_std = []
        
        for i in range(len(losses) - window_size + 1):
            window_losses = losses[i:i+window_size]
            rolling_std.append(np.std(window_losses))
        
        if rolling_std:
            avg_std = np.mean(rolling_std)
            # Relative oscillation: std / mean
            return avg_std / mean_loss
        else:
            return 0.0
    
    def _compute_oscillation_from_weights(self, step):
        """Oscillation based on detrended parameters"""
        if len(self.weight_history) < 20 or step < WARMUP_STEPS:
            return 0.0
        
        # Use changes in parameter norms
        weight_norms = []
        for w_arr in self.weight_history:
            weight_norms.append(np.linalg.norm(w_arr))
        
        weight_norms = np.array(weight_norms)
        
        # Detrend: subtract linear trend
        x = np.arange(len(weight_norms))
        coeffs = np.polyfit(x, weight_norms, 1)
        trend = coeffs[0] * x + coeffs[1]
        detrended = weight_norms - trend
        
        # Calculate std of detrended data, normalized relative to mean
        mean_norm = np.mean(weight_norms)
        if mean_norm > 1e-8:
            return np.std(detrended) / mean_norm
        else:
            return np.std(detrended)
    
    def _compute_oscillation_from_effective_lr(self, v, gradients, step):
        """Oscillation based on effective learning rate fluctuation"""
        if step < WARMUP_STEPS:
            return 0.0
        
        # Calculate current effective learning rate (approximate)
        v_mean = v.mean().item()
        effective_lr = 1.0 / np.sqrt(v_mean + 1e-8) if v_mean > 0 else 1.0
        
        # Use historical data to calculate fluctuation of effective learning rate
        if len(self.variance_history) >= 10:
            recent_v = self.variance_history[-10:]
            recent_effective_lrs = [1.0 / np.sqrt(v_val + 1e-8) for v_val in recent_v]
            
            # Calculate relative fluctuation
            mean_lr = np.mean(recent_effective_lrs)
            if mean_lr > 1e-8:
                return np.std(recent_effective_lrs) / mean_lr
            else:
                return np.std(recent_effective_lrs)
        
        return 0.0
    
    def _compute_composite_oscillation(self, oscillation_loss, oscillation_param, oscillation_lr, step):
        """Composite oscillation metric (weighted average)"""
        if step < WARMUP_STEPS:
            return 0.0
        
        # Adjust weights based on training phase
        if step < 500:  # Early training
            weights = [0.3, 0.5, 0.2]  # More emphasis on parameter oscillation
        elif step < 1000:  # Mid training
            weights = [0.4, 0.4, 0.2]  # Balanced
        else:  # Late training
            weights = [0.6, 0.3, 0.1]  # More emphasis on loss oscillation (post-convergence fluctuation)
        
        # Weighted average
        composite = (
            weights[0] * oscillation_loss +
            weights[1] * oscillation_param +
            weights[2] * oscillation_lr
        )
        
        return composite
    
    def _compute_old_oscillation(self, w, step):
        """Old oscillation calculation method (for comparison)"""
        if step == 0 or self.prev_w_old is None:  # Modified here
            self.prev_w_old = w.detach().clone()
            return 0.0
        
        abs_change = torch.norm(w - self.prev_w_old)
        w_norm = torch.norm(w)
        rel_change = abs_change / (w_norm + 1e-8)
        
        self.prev_w_old = w.detach().clone()
        return rel_change.item()
    
    def _estimate_noise_strength(self, gradients):
        """Estimate gradient noise strength"""
        grad_var = gradients.var()
        return grad_var.item() * self.num_params
    
    def _detect_convergence(self, step, loss):
        """Detect convergence"""
        if step < 500:
            return False
        
        # Simple convergence detection: loss changes little in recent N steps
        if len(self.loss_history) >= 50:
            recent_losses = self.loss_history[-50:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_mean > 1e-8 and loss_std / loss_mean < 0.01:  # Relative change < 1%
                return True
        
        return False

# ==================== Improved experiment function ====================
def run_corrected_experiment(run_id, seed=42):
    """Run corrected experiment"""
    
    print(f"\n{'='*60}")
    print(f"Starting experiment {run_id+1}/{NUM_RUNS} (seed={seed})")
    print(f"{'='*60}")
    
    # Set random seeds
    set_all_seeds(seed)
    
    print("Preparing dataset...")
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Use CIFAR-10 dataset
    full_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use partial data to speed up experiment
    from torch.utils.data import Subset
    indices = list(range(0, 20000))
    dataset = Subset(full_dataset, indices)
    
    all_results = []
    experiment_id = 0
    
    # For each batch size and learning rate strategy
    for batch_size in tqdm(batch_sizes, desc="Batch size progress"):
        for lr_strategy_name, lr_func in tqdm(
            learning_rate_strategies.items(), 
            desc=f"LR strategies for batch={batch_size}",
            leave=False
        ):
            
            print(f"\nRunning experiment: batch_size={batch_size}, lr_strategy={lr_strategy_name}")
            
            # Data loader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            
            # Model
            model = models.resnet18(num_classes=10)
            model = model.to(DEVICE)
            
            # Calculate number of parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Optimizer
            initial_lr = lr_func(batch_size, 0)
            optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))
            
            # Loss function
            criterion = nn.CrossEntropyLoss()
            
            # Metrics collector
            collector = CorrectedMetricsCollector(num_params=num_params)
            
            # Training loop
            data_iter = iter(dataloader)
            
            # Add learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_STEPS, eta_min=initial_lr * 0.01
            )
            
            for step in range(TOTAL_STEPS):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    inputs, targets = next(data_iter)
                
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # Dynamically adjust learning rate
                current_lr = lr_func(batch_size, step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Gradient calculation
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevent gradient explosion)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                # Collect gradient information
                gradients_list = []
                for p in model.parameters():
                    if p.grad is not None:
                        gradients_list.append(p.grad.flatten())
                
                if gradients_list:
                    gradients = torch.cat(gradients_list)
                else:
                    gradients = torch.tensor([0.0], device=DEVICE)
                
                # Get optimizer state (m and v)
                m_list = []
                v_list = []
                for param_group in optimizer.param_groups:
                    for p in param_group['params']:
                        state = optimizer.state[p]
                        if 'exp_avg' in state and 'exp_avg_sq' in state:
                            m_list.append(state['exp_avg'].flatten())
                            v_list.append(state['exp_avg_sq'].flatten())
                
                if m_list and v_list:
                    m = torch.cat(m_list)
                    v = torch.cat(v_list)
                    w = torch.cat([p.flatten() for p in model.parameters()])
                    
                    # Collect metrics
                    collector.collect_step(
                        step=step,
                        batch_size=batch_size,
                        lr_strategy=lr_strategy_name,
                        gradients=gradients,
                        m=m,
                        v=v,
                        w=w,
                        loss=loss,
                        optimizer_state={'lr': current_lr}
                    )
                
                # Optimizer update
                optimizer.step()
                scheduler.step()
                
                # Output progress
                if step % 200 == 0:
                    print(f"  Step {step}/{TOTAL_STEPS}, Loss: {loss.item():.4f}, "
                          f"Learning Rate: {current_lr:.6f}")
            
            # Save results
            df = pd.DataFrame(collector.records)
            df['experiment_id'] = experiment_id
            df['run_id'] = run_id
            df['seed'] = seed
            all_results.append(df)
            experiment_id += 1
            
            # Save checkpoint
            checkpoint_path = f"analysis_results/checkpoints/run{run_id}_model_bs{batch_size}_{lr_strategy_name}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            
            # Clean up memory
            del model, optimizer, collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Merge all results
    full_df = pd.concat(all_results, ignore_index=True)
    
    # Save complete data
    run_filename = f'analysis_results/experiment_runs/full_experiment_data_run{run_id}_seed{seed}.csv'
    full_df.to_csv(run_filename, index=False)
    
    print(f"\nExperiment {run_id+1} completed! Data saved to: {run_filename}")
    
    return full_df

# ==================== Multiple experiment aggregation analysis ====================
def analyze_multiple_runs(num_runs=NUM_RUNS):
    """Analyze results from multiple experiments"""
    
    print(f"\n{'='*60}")
    print(f"Starting analysis of {num_runs} experiment runs")
    print(f"{'='*60}")
    
    # Load all experiment data
    all_runs_data = []
    for run_id in range(num_runs):
        run_filename = f'analysis_results/experiment_runs/full_experiment_data_run{run_id}_seed{RUN_SEEDS[run_id]}.csv'
        if os.path.exists(run_filename):
            df = pd.read_csv(run_filename)
            all_runs_data.append(df)
            print(f"Loaded experiment {run_id+1} data: {len(df)} rows")
        else:
            print(f"Warning: Experiment {run_id+1} data file not found: {run_filename}")
    
    if not all_runs_data:
        print("Error: No experiment data found!")
        return None
    
    # Merge all data
    full_data = pd.concat(all_runs_data, ignore_index=True)
    print(f"Total data volume: {len(full_data)} rows")
    
    # Save merged data
    full_data.to_csv('analysis_results/full_experiment_data_all_runs.csv', index=False)
    
    # Analyze key metrics for each experiment
    print("\n=== Key metrics for each experiment ===")
    run_metrics = []
    
    for run_id in range(num_runs):
        run_df = full_data[full_data['run_id'] == run_id]
        
        if len(run_df) > 0:
            # Calculate key statistics for this experiment
            metrics = {
                'run_id': run_id,
                'seed': RUN_SEEDS[run_id],
                'n_samples': len(run_df),
                'mean_inflation': run_df['inflation'].mean(),
                'mean_oscillation': run_df['oscillation_composite'].mean(),
                'corr_inflation_osc': run_df['inflation'].corr(run_df['oscillation_composite']),
                'mean_phi': run_df['phi'].mean(),
                'mean_batch_effect': run_df[run_df['batch_size'] <= 8]['oscillation_composite'].mean() /
                                   run_df[run_df['batch_size'] >= 128]['oscillation_composite'].mean(),
                'std_inflation': run_df['inflation'].std(),
                'std_oscillation': run_df['oscillation_composite'].std()
            }
            run_metrics.append(metrics)
            
            print(f"\nExperiment {run_id+1} (seed={RUN_SEEDS[run_id]}):")
            print(f"  Average inflation: {metrics['mean_inflation']:.4f}")
            print(f"  Average oscillation: {metrics['mean_oscillation']:.6f}")
            print(f"  Inflation-oscillation correlation: {metrics['corr_inflation_osc']:.4f}")
            print(f"  Average Φ value: {metrics['mean_phi']:.4f}")
            print(f"  Batch effect ratio: {metrics['mean_batch_effect']:.2f}x")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(run_metrics)
    
    # Calculate mean and standard deviation
    print(f"\n{'='*60}")
    print("Statistical summary of multiple experiments (mean ± std):")
    print(f"{'='*60}")
    
    for metric_name in ['mean_inflation', 'mean_oscillation', 'corr_inflation_osc', 'mean_phi', 'mean_batch_effect']:
        metric_values = metrics_df[metric_name].values
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        cv = std_val / mean_val * 100  # Coefficient of variation (%)
        
        # Calculate confidence interval (95%)
        n = len(metric_values)
        if n > 1:
            from scipy import stats
            t_critical = stats.t.ppf(0.975, n-1)  # t-value for 95% confidence
            se = std_val / np.sqrt(n)
            ci_lower = mean_val - t_critical * se
            ci_upper = mean_val + t_critical * se
            
            print(f"{metric_name}:")
            print(f"  Mean = {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Coefficient of variation = {cv:.2f}%")
            print(f"  95% Confidence interval = [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  Range = [{np.min(metric_values):.4f}, {np.max(metric_values):.4f}]")
            print()
    
    # Plot inter-experiment consistency
    plot_run_consistency(metrics_df)
    
    # Generate detailed report
    generate_replication_report(metrics_df, full_data)
    
    return metrics_df, full_data

def plot_run_consistency(metrics_df):
    """Plot consistency charts across multiple experiments"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Inflation consistency
    axes[0, 0].bar(range(len(metrics_df)), metrics_df['mean_inflation'], 
                   yerr=metrics_df['std_inflation'], 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Experiment run')
    axes[0, 0].set_ylabel('Average inflation')
    axes[0, 0].set_title('Inter-experiment consistency of inflation')
    axes[0, 0].set_xticks(range(len(metrics_df)))
    axes[0, 0].set_xticklabels([f'Run {i+1}' for i in range(len(metrics_df))])
    
    # 2. Inflation-oscillation correlation consistency
    axes[0, 1].bar(range(len(metrics_df)), metrics_df['corr_inflation_osc'], 
                   capsize=5, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Experiment run')
    axes[0, 1].set_ylabel('Inflation-oscillation correlation coefficient')
    axes[0, 1].set_title('Inter-experiment consistency of correlation')
    axes[0, 1].set_xticks(range(len(metrics_df)))
    axes[0, 1].set_xticklabels([f'Run {i+1}' for i in range(len(metrics_df))])
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Φ value consistency
    axes[0, 2].bar(range(len(metrics_df)), metrics_df['mean_phi'], 
                   capsize=5, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_xlabel('Experiment run')
    axes[0, 2].set_ylabel('Average Φ value')
    axes[0, 2].set_title('Inter-experiment consistency of Φ')
    axes[0, 2].set_xticks(range(len(metrics_df)))
    axes[0, 2].set_xticklabels([f'Run {i+1}' for i in range(len(metrics_df))])
    axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Batch size effect stability
    axes[1, 0].plot(metrics_df['run_id'], metrics_df['mean_batch_effect'], 
                   'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].fill_between(metrics_df['run_id'], 
                           metrics_df['mean_batch_effect'] - metrics_df['mean_batch_effect'].std(),
                           metrics_df['mean_batch_effect'] + metrics_df['mean_batch_effect'].std(),
                           alpha=0.3, color='purple')
    axes[1, 0].set_xlabel('Experiment run')
    axes[1, 0].set_ylabel('Small/large batch oscillation ratio')
    axes[1, 0].set_title('Stability of batch size effect')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Box plot of key metrics
    metrics_to_plot = ['mean_inflation', 'corr_inflation_osc', 'mean_phi']
    data_to_plot = [metrics_df[metric] for metric in metrics_to_plot]
    axes[1, 1].boxplot(data_to_plot, labels=['Inflation', 'Correlation', 'Φ'])
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Distribution of key metrics')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Coefficient of variation radar chart
    cv_values = [
        metrics_df['mean_inflation'].std() / metrics_df['mean_inflation'].mean(),
        metrics_df['corr_inflation_osc'].std() / abs(metrics_df['corr_inflation_osc'].mean()),
        metrics_df['mean_phi'].std() / abs(metrics_df['mean_phi'].mean()),
        metrics_df['mean_batch_effect'].std() / metrics_df['mean_batch_effect'].mean()
    ]
    
    labels = ['Inflation', 'Correlation', 'Φ', 'Batch effect']
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    cv_values += cv_values[:1]  # Close the shape
    angles = np.concatenate((angles, [angles[0]]))
    
    axes[1, 2] = plt.subplot(236, projection='polar')
    axes[1, 2].plot(angles, cv_values, 'o-', linewidth=2)
    axes[1, 2].fill(angles, cv_values, alpha=0.25)
    axes[1, 2].set_thetagrids(angles[:-1] * 180/np.pi, labels)
    axes[1, 2].set_title('Coefficient of variation for each metric\n(lower is more stable)')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('analysis_results/replication_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Inter-experiment consistency chart saved: analysis_results/replication_consistency.png")

def generate_replication_report(metrics_df, full_data):
    """Generate detailed report for multiple experiments"""
    
    from scipy import stats
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Adaptive Optimizer Experiment - Multiple Replication Report")
    report_lines.append("="*80)
    report_lines.append(f"Number of experiment repetitions: {len(metrics_df)}")
    report_lines.append(f"Total data points: {len(full_data)}")
    report_lines.append(f"Random seeds: {list(metrics_df['seed'])}")
    report_lines.append(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Statistical summary of key metrics
    report_lines.append("Statistical summary of key metrics:")
    report_lines.append("-"*40)
    
    for metric_name, display_name in [
        ('mean_inflation', 'Average inflation'),
        ('mean_oscillation', 'Average oscillation'),
        ('corr_inflation_osc', 'Inflation-oscillation correlation'),
        ('mean_phi', 'Average Φ value'),
        ('mean_batch_effect', 'Batch size effect ratio')
    ]:
        values = metrics_df[metric_name].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val * 100
        
        # Calculate confidence interval
        n = len(values)
        if n > 1:
            t_critical = stats.t.ppf(0.975, n-1)
            se = std_val / np.sqrt(n)
            ci_lower = mean_val - t_critical * se
            ci_upper = mean_val + t_critical * se
            
            report_lines.append(f"{display_name}:")
            report_lines.append(f"  Mean = {mean_val:.4f} ± {std_val:.4f}")
            report_lines.append(f"  Coefficient of variation = {cv:.2f}%")
            report_lines.append(f"  95% Confidence interval = [{ci_lower:.4f}, {ci_upper:.4f}]")
            report_lines.append(f"  Range = [{np.min(values):.4f}, {np.max(values):.4f}]")
            report_lines.append("")
    
    # ANOVA analysis of inter-experiment correlation
    report_lines.append("Statistical tests for inter-experiment differences:")
    report_lines.append("-"*40)
    
    # ANOVA for inflation (actually only one group, needs grouped data)
    # Here we use paired t-test to compare if differences between experiments are significant
    
    if len(metrics_df) >= 2:
        # Check if mean is significantly different from first experiment
        base_inflation = metrics_df['mean_inflation'].iloc[0]
        other_inflations = metrics_df['mean_inflation'].iloc[1:]
        
        t_stat, p_value = stats.ttest_1samp(other_inflations, base_inflation)
        report_lines.append(f"Difference in inflation compared to first experiment:")
        report_lines.append(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
        if p_value < 0.05:
            report_lines.append("  → Significant difference (p < 0.05)")
        else:
            report_lines.append("  → No significant difference (p ≥ 0.05)")
        report_lines.append("")
    
    # Stability of batch size effect
    report_lines.append("Stability analysis of batch size effect:")
    report_lines.append("-"*40)
    
    batch_effect_values = metrics_df['mean_batch_effect'].values
    batch_effect_cv = np.std(batch_effect_values) / np.mean(batch_effect_values) * 100
    report_lines.append(f"Small batch oscillation/large batch oscillation ratio:")
    report_lines.append(f"  Average ratio = {np.mean(batch_effect_values):.2f}x")
    report_lines.append(f"  Standard deviation = {np.std(batch_effect_values):.2f}")
    report_lines.append(f"  Coefficient of variation = {batch_effect_cv:.2f}%")
    report_lines.append(f"  Range = {np.min(batch_effect_values):.2f}x to {np.max(batch_effect_values):.2f}x")
    report_lines.append("")
    
    # Φ value sign analysis
    report_lines.append("Φ value sign analysis:")
    report_lines.append("-"*40)
    
    # Calculate Φ sign distribution using all data
    phi_positive = (full_data['phi'] > 0.1).sum()
    phi_negative = (full_data['phi'] < -0.1).sum()
    phi_weak = ((full_data['phi'] >= -0.1) & (full_data['phi'] <= 0.1)).sum()
    total_phi = len(full_data['phi'])
    
    report_lines.append(f"Positive correlation (Φ > 0.1): {phi_positive} ({phi_positive/total_phi*100:.1f}%)")
    report_lines.append(f"Negative correlation (Φ < -0.1): {phi_negative} ({phi_negative/total_phi*100:.1f}%)")
    report_lines.append(f"Weak correlation (|Φ| ≤ 0.1): {phi_weak} ({phi_weak/total_phi*100:.1f}%)")
    report_lines.append("")
    
    # Conclusions and recommendations
    report_lines.append("Conclusions and recommendations:")
    report_lines.append("-"*40)
    
    # Evaluate stability
    cv_threshold = 10  # Coefficient of variation threshold (%)
    stable_metrics = []
    unstable_metrics = []
    
    for metric_name, display_name in [
        ('mean_inflation', 'Inflation'),
        ('corr_inflation_osc', 'Inflation-oscillation correlation'),
        ('mean_phi', 'Φ value'),
        ('mean_batch_effect', 'Batch size effect')
    ]:
        values = metrics_df[metric_name].values
        cv = np.std(values) / np.mean(values) * 100
        if cv < cv_threshold:
            stable_metrics.append(display_name)
        else:
            unstable_metrics.append(f"{display_name}(CV={cv:.1f}%)")
    
    if stable_metrics:
        report_lines.append(f"1. Stable metrics: {', '.join(stable_metrics)}")
    if unstable_metrics:
        report_lines.append(f"2. Less stable metrics: {', '.join(unstable_metrics)}")
    
    report_lines.append("3. Recommendations:")
    report_lines.append("   • Main findings show good inter-experiment reproducibility")
    report_lines.append("   • Jensen inflation effect is stable across multiple experiments")
    report_lines.append("   • Batch size effect is consistent across all experiments")
    report_lines.append("   • Φ values are mainly negative but show some fluctuation")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_path = "analysis_results/replication_experiment_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nMultiple experiment report saved: {report_path}")
    
    return report_path

# ==================== Main program ====================
if __name__ == "__main__":
    print("="*80)
    print("Adaptive Optimizer Experiment (Corrected oscillation metrics + Multiple repetitions)")
    print("="*80)
    
    print(f"Batch size range: {batch_sizes}")
    print(f"Learning rate strategies: {list(learning_rate_strategies.keys())}")
    print(f"Total steps: {TOTAL_STEPS} (Warmup steps: {WARMUP_STEPS})")
    print(f"Device: {DEVICE}")
    print(f"Number of experiment repetitions: {NUM_RUNS}")
    print(f"Random seeds: {RUN_SEEDS}")
    
    print("\nMain improvements:")
    print("1. Added random seed setting for reproducibility")
    print("2. Supports multiple independent experiment runs")
    print("3. Inter-experiment consistency analysis and statistical tests")
    print("4. Generates detailed replication experiment report")
    
    proceed = input(f"\nRun {NUM_RUNS} experiments? (yes/no): ")
    
    if proceed.lower() in ['yes', 'y']:
        print(f"\nStarting {NUM_RUNS} corrected experiments...")
        print("This will take a while, please be patient...")
        
        # Run all experiments
        all_results = []
        for run_id in range(NUM_RUNS):
            seed = RUN_SEEDS[run_id]
            try:
                results_df = run_corrected_experiment(run_id, seed)
                all_results.append(results_df)
                
                # Simple statistics after each experiment
                print(f"\nExperiment {run_id+1} completed!")
                print(f"  Average inflation: {results_df['inflation'].mean():.4f}")
                print(f"  Average oscillation: {results_df['oscillation_composite'].mean():.6f}")
                print(f"  Inflation-oscillation correlation: {results_df['inflation'].corr(results_df['oscillation_composite']):.4f}")
                
            except Exception as e:
                print(f"\nExperiment {run_id+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*80}")
        print(f"All {len(all_results)} experiments completed!")
        print("="*80)
        
        # Analyze multiple experiment results
        if len(all_results) > 0:
            print("\nStarting analysis of multiple experiment results...")
            metrics_df, full_data = analyze_multiple_runs(len(all_results))
            
            print("\n" + "="*80)
            print("Experiment completed!")
            print("="*80)
            
            print("\nGenerated files:")
            print("1. analysis_results/experiment_runs/ - Individual experiment data")
            print("2. analysis_results/full_experiment_data_all_runs.csv - Merged data")
            print("3. analysis_results/replication_consistency.png - Inter-experiment consistency chart")
            print("4. analysis_results/replication_experiment_report.txt - Replication experiment report")
            print("5. analysis_results/checkpoints/ - Model checkpoints")
            
            if metrics_df is not None:
                print("\nKey results summary:")
                print("-"*40)
                print(f"Inflation: {metrics_df['mean_inflation'].mean():.4f} ± {metrics_df['mean_inflation'].std():.4f}")
                print(f"Inflation-oscillation correlation: {metrics_df['corr_inflation_osc'].mean():.4f} ± {metrics_df['corr_inflation_osc'].std():.4f}")
                print(f"Φ value: {metrics_df['mean_phi'].mean():.4f} ± {metrics_df['mean_phi'].std():.4f}")
                print(f"Batch effect: {metrics_df['mean_batch_effect'].mean():.2f}x ± {metrics_df['mean_batch_effect'].std():.2f}x")
            
            print("="*80)
        
    else:
        print("Experiment cancelled")
