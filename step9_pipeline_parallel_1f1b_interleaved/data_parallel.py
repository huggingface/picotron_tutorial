import contextlib
from typing import List
import torch
import torch.distributed as dist
from torch import nn

import process_group_manager as pgm

### begin Data Parallel (naive)
class DataParallelNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.require_backward_grad_sync = True
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """Registers a backward hook for all parameters of the model that require gradients.""" 
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        """Performs an all-reduce operation to synchronize gradients across multiple processes."""
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.dp_group)
            grad /= pgm.process_group_manager.dp_world_size
        return grad

### end Data Parallel (naive)

### begin Data Parallel (bucket)
class Bucket:
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        # Set of parameters in this bucket.
        self.params = set(params)
        # Parameters that have their gradients ready for synchronization. launch all reduce when all parameters have gradients ready
        self.params_with_grad_ready = set()
        # Parameters that have their gradients ready for synchronization. launch all reduce when all parameters have gradients ready
        self.grad_data = grad_data
        # Process group for gradient synchronization.
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group) 
        # Handle for the async allreduce operation.
        self.handle = None
        
        self.reset()
    
    def sync_gradient(self) -> None:
        """Launch an asynchronous all-reduce operation to synchronize gradients across processes."""
        assert self.handle is None
        self.grad_data /= self.process_group_size
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)
    
    def reset(self) -> None:
        """Reset the bucket to its initial state. Typically called after the gradient synchronization is finished."""
        self.handle = None
        # Clear the set of parameters ready for gradient synchronization.
        self.params_with_grad_ready.clear()
        # Zero the gradient tensor.
        self.grad_data.zero_()

    def wait(self) -> None:
        """wait for the allreduce operation to finish"""
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        # Block until the all-reduce operation finishes.
        self.handle.wait()

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        """Mark a parameter as ready for gradient synchronization. Launches synchronization when all parameters in the bucket have their gradients ready."""
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        # When all parameters in the bucket have their gradients ready, synchronize gradients
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

class BucketManager:
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:
        # Convert parameter generator to a list.
        self.params = list(params)
        # List of buckets.
        self.buckets = []
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        # Map each parameter to its corresponding bucket/place (start, end, bucket_idx).
        self.params_to_bucket_location = {}
        # Actual sizes of each bucket.
        self.bucket_size = bucket_size
        self.bucket_sizes = None
        # List of tensors to store gradients, one tensor per bucket.
        self.grad_data_list = []
        self.grad_type = grad_type
        # Divide gradients into buckets based on the provided bucket size.
        self._initialize_buckets()
    
    def _initialize_buckets(self) -> None:
        """Divides model parameters into buckets for gradient synchronization based on the bucket size."""
        cur_bucket_size = 0 
        cur_bucket_idx = 0
        
        # Assign parameters to buckets. 
        for param in self.params:
            if not param.requires_grad:
                continue
            
            # If the bucket is empty, add the parameter to the bucket.
            if cur_bucket_size == 0:
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
                continue
            
            # If the parameter cannot fit in the current bucket, create a new bucket
            if cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
            else:
                self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
                cur_bucket_size += param.numel()

        # Gather information about the bucket sizes and the parameters in each bucket
        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)
        
        # Create tensors for storing gradients and initialize Bucket objects.
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device='cuda'))
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))
        
        # Create gradient views for each parameter.
        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            # param.main_grad is used for gradient calculation
            param.main_grad = self._get_view_from_tensor(self.grad_data_list[bucket_id], param.shape, data_start_index, data_end_index)
            
    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int) -> torch.Tensor:
        return tensor[start:end].view(shape)
    
    def reset(self) -> None:
        # Reset all buckets by clearing the gradients and internal states.
        for bucket in self.buckets:
            bucket.reset()
    
    def wait(self) -> None:
        # Wait for all buckets to complete their gradient synchronization
        for bucket in self.buckets:
            bucket.wait()
    
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        # Mark a parameter's gradient as ready for synchronization.
        bucket_idx = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)

class DataParallelBucket(nn.Module):
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        super().__init__()
        self.module = module
        # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.require_backward_grad_sync = True
        grad_size = 2 # bfloat16 gradient: 2 bytes
        # number of gradients in one bucket
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        # whether the callback for wait gradient synchronization is set
        self._post_backward_callback_set = False
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def get_flops(self, *args, **kwargs):
        return self.module.get_flops(*args, **kwargs)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        torch.autograd.Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook

    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        self.bucket_manager.reset()

### end Data Parallel (bucket)