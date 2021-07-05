# Weight Pruning via Adaptive Sparsity Loss

This repository contains the code for pruning any given architecture according to a sparsity percentage target, as proposed in the following paper: *"Weight Pruning via Adaptive Sparsity Loss" [[arxiv]](https://arxiv.org/abs/2006.02768)*

The main idea is to introduce a single trainable parameter per layer which controls the sparsity of the layer. This extra parameter acts as the threshold of a pruning operation (any weight under this threshold is pruned) and is optimized with respect to a multi-task loss consisted of the task loss and the sparsity controlling loss (user provides a requested target sparsity). 

The sparsity controlling loss, the main novelty of *"Weight Pruning via Adaptive Sparsity Loss"*, relies on the assumption of a Gaussian distribution over the weights at each layer.  Such assumption, retained by the application of Straight Through Estimator [[arxiv]](https://arxiv.org/abs/1308.3432) at each pruning operation, enables us to formulate the sparsity at each layer as an analytic function w.r.t to first order statistics (mean value and standard deviation) and the trainable pruning parameter, using the *erf* function. For a detailed description of the adaptive sparsity loss formulation, see the paper [[arxiv]](https://arxiv.org/abs/2006.02768).

<!---***sparsity:*** $s  = \text{erf}(\frac{b}{\sigma \sqrt{2}})$ ,  where  $\text{erf}(x) = \frac{1}{\sqrt{\pi}}\int_{-x}^{x}e^{-t^2}dt$     *(error function)*--->

<!---***pruning function:*** $f_{prune}(w; b) =
    \begin{cases}
      0, & \text{if}\ |w| < b \\
      w, & \text{otherwise}
    \end{cases}$--->
    
The sparsity loss can be formulated according to the user's needs (see paper) and the basic tools for sparsifying any architecture are provided at *sparse_utils.py*. An example of using these sparsity tools is also provided for the setting of Wide ResNets [[arxiv]](https://arxiv.org/abs/1605.07146) and the CIFAR100 dataset.


Files: 
 - sparse_example.py (example script: WRNet-16-8 & CIFAR100)
 - wide_resnet.py (Wide ResNet implementation)
 - sparse_utils.py (sparsity tools/ auxiliary functions)

**Tested on PyTorch 1.3 (torch, torchvision & scipy packages are required)**

-------------------------------------------------------------------------

### Sparsification Options of *sparse_example.py* ( --sparsity / --pthres / --starget / --lv)

**Sparsification Method** (--sparsity fixed/adaptive): 
- fixed: each layer has a predefined fixed sparsity percentage
- adaptive: seek the model's sparsity (as weighted sum of per layer sparsity) according a overall budget constraint.

**Sparsification Hyper-parameters:**
- starget: the requested sparsity (e.g. .9 for 90% sparsity). Used for fixed/budget alternatives.
- lv: the sparsity loss weight. Controls the balance between task loss and sparsity loss (*task_loss* + lv * *sparsity_loss*). 

**Minimum Layer Size:**  Layers that exceed a predefined number of parameters are sparsified ( --pthres, default value:1000) 

-------------------------------------------------------------------------

### Training Options of *sparse_example.py* ( --epochs / --lr / --batch-size/ --gpu)

- epochs: number of overall epochs (default: 60)
- lr: initial learning rate (default: 0.1)
- batch-size: input batch size for training (default: 128)
- gpu: select GPU device (by id, e.g. 0)

(existing scheduler is Cosine Annealing with warm restarts - 1 restart @ epochs/2)

-------------------------------------------------------------------------

### Examples:

    python3 sparse_example.py --gpu 0 --sparsity adaptive --starget .9 --lv 10.0

-------------------------------------------------------------------------
### Sparsity inducing functions of *sparse_utils.py*:

The file *sparse_utils.py* contains the pruning-related functions that can be applied at any architecture.

 - iter_sparsify(model, thresh, trainable=True, pthres=1000): 	 sparsify (iteratively) every convolutional of fully connected layer with more than *pthres* parameters.  This function can sparsify such layers for any given architecture. The initial pruning threshold *thres* is provided as input, along with the boolean property of training this threshold (if is set to False, the sparsity level is retained through training).
 - iter_desparsify(model): remove the extra threshold parameters and returns the sparsified model (used after the training procedure).
 - sparsity(model, print_per_layer=False): computes and prints the per layer and the overall sparsity of the network.
 - adaptive_loss(model, reduce=True): returns the sparsity at each layer (*reduce=False*). When the *reduce* is set to True, an overall sparsity loss is computed w.r.t. the parameters of the network (weighted average of the per layer sparsities). 

-------------------------------------------------------------------------
### Citing:

```
@inproceedings{retsinas2021weight,
  title={Weight Pruning via Adaptive Sparsity Loss},
  author={Retsinas, George and Elafrou, Athena and Goumas, Georgios and Maragos, Petros},
  booktitle={2021 IEEE international conference on image processing (ICIP)},
  year={2021},
  organization={IEEE}
}
```
