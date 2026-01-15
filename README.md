# jensen-inflation-phi-effect


🔬 Experimental validation of Jensen inflation and Φ effect in adaptive optimizers (Adam/AdamW)

## Research Focus
- **Jensen Inflation**: Analysis of second moment estimation bias in adaptive optimizers
- **Φ Effect**: Correlation between gradient noise and its square (ξ ~ ξ²)
- **Batch Size Effects**: How batch size impacts optimization stability and convergence
- **Learning Rate Scaling**: Comparison of fixed, linear, and sqrt scaling strategies

## Key Features
 Multiple experiment repetitions (3 runs with different seeds)  
 Statistical analysis with confidence intervals  
 Corrected oscillation metrics calculation  
 Comprehensive metrics collection (gradient stats, inflation, Φ, oscillations)  
 Visualization of inter-experiment consistency  
 Replication study methodology

## Technical Details
- **Model**: ResNet-18 on CIFAR-10 dataset
- **Batch Sizes**: [2, 4, 8, 16, 32, 64, 128, 256]
- **Learning Rate Strategies**: Fixed, Linear scaling, sqrt scaling
- **Metrics**: Gradient statistics, tail index, inflation rate, Φ value, oscillation metrics
- **Framework**: PyTorch with extensive logging and analysis

## Outputs
- Statistical validation of theoretical predictions
- Replication consistency analysis
- Detailed experiment reports
- Visualizations and checkpoints

## Related Topics
#AdaptiveOptimizers #AdamOptimizer #JensenInequality #BatchSizeEffects #OptimizationTheory #DeepLearningResearch #EmpiricalStudy #ReplicationStudy
