import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import process_group_manager as pgm

from typing import List

### begin PP communications
STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"
def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    """
    Handles point-to-point communication between pipeline stages for forward and backward passes.
    
    Args:
        operation (str): Type of communication operation ('recv_forward', 'send_forward', 
                        'recv_backward', 'send_backward')
        device: Target device for tensor operations (e.g., CPU, GPU)
        dtype: Data type for tensors
        tensor: Input tensor for send operations (default: None)
        shapes: Shape specifications for receiving tensors (default: None)
    
    Returns:
        torch.Tensor or None: Received tensor for receive operations, None for send operations
    """
    global STEP
    global VERBOSE
    
    if operation == 'recv_forward':
        # Skip if this is the first pipeline stage (nothing to receive)
        if pgm.process_group_manager.pp_is_first_stage: return None
        # Create empty tensor to receive data
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
    
    elif operation == 'send_forward':
        # Skip if this is the last pipeline stage (nothing to send forward)
        if pgm.process_group_manager.pp_is_last_stage: return
        dest = pgm.process_group_manager.pp_next_rank
    
    elif operation == 'recv_backward':
        # Skip if this is the last pipeline stage (nothing to receive from backward)
        if pgm.process_group_manager.pp_is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
    
    elif operation == 'send_backward':
        # Skip if this is the first pipeline stage (nothing to send backward)
        if pgm.process_group_manager.pp_is_first_stage: return
        dest = pgm.process_group_manager.pp_prev_rank

    # Determine if this is a send operation and set peer rank
    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src
    
    # Create P2P operation (send or receive)
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    
    if VERBOSE: 
        print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} "
              f"{pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | "
              f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    
    # Execute communication operation and wait for completion
    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()
    
    if VERBOSE: STEP += 1
    
    # Return received tensor for receive operations, None for send operations
    return tensor if not is_send else None

def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    """
    Handles bidirectional communication between pipeline stages, allowing simultaneous 
    send and receive operations.
    
    Args:
        operation (str): Type of bidirectional operation ('send_fwd_recv_bwd' or 'send_bwd_recv_fwd')
        send_tensor: Tensor to be sent
        recv_shapes: Shape specifications for the tensor to be received
        device: Target device for tensor operations
        dtype: Data type for tensors
    
    Returns:
        torch.Tensor or None: Received tensor, or None if at terminal pipeline stage
    """
    global STEP
    global VERBOSE
    
    # Determine if this is a forward operation
    is_fwd = (operation == 'send_fwd_recv_bwd')
    
    # Skip if at terminal pipeline stages
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or \
       (not is_fwd and pgm.process_group_manager.pp_is_first_stage): 
        return None
    
    # Determine peer rank based on operation direction
    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    
    # Create empty tensor for receiving data
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)
    
    # Set up simultaneous send and receive operations
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send_tensor, peer_rank),
        dist.P2POp(dist.irecv, recv_tensor, peer_rank)
    ])
    
    if VERBOSE: 
        print(f"{operation} | sending {'next' if is_fwd else 'prev'} "
              f"{pgm.process_group_manager.pp_rank} -> {peer_rank} | "
              f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> "
              f"{pgm.process_group_manager.pp_rank} | STEP {STEP=} | "
              f"RANK:{pgm.process_group_manager.pp_rank}", flush=True)
    
    # Wait for both operations to complete
    [req.wait() for req in reqs]
    torch.cuda.synchronize()
    
    if VERBOSE: STEP += 1
    
    return recv_tensor
### end PP communications

### begin Pipeline Parallel
class PipelineParallel(nn.Module):
    def __init__(self, model, config, pp_engine, num_virtual_pipeline_stages):
        super().__init__()
        
        self.pp_engine = pp_engine
        self.embedding = model.embedding if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        
        if pp_engine == "1f1b_interleaved":
            assert num_virtual_pipeline_stages > 0; "Invalid configuration for 1F1B interleaved pipeline. Number of virtual pipeline stages must be > 0."
            layer_distribution = self.distribute_layers_interleaved(config.num_hidden_layers, num_virtual_pipeline_stages)
            self.decoder_layers = []
            for i in range(len(layer_distribution)):
                self.decoder_layers.append(
                    nn.ModuleDict({str(i): model.decoder_layers[j] for j in layer_distribution[i]})
                )
            #TODO: Understand how Megatron deal with different virtual stages indexation. This influence forward pass
        else:
            assert self.pp_engine in ["1f1b", "afab"]
            layer_distribution = self.distribute_layers(config.num_hidden_layers)
            self.decoder_layers = nn.ModuleDict({str(i): model.decoder_layers[i] for i in layer_distribution})

        self.final_norm = model.final_norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.final_proj = model.final_proj if pgm.process_group_manager.pp_is_last_stage else nn.Identity()

    def distribute_layers(self, num_layers: int) -> List[List[int]]:
        num_layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) for i in range(pgm.process_group_manager.pp_world_size)]
        start_layer = sum(num_layers_per_gpu[:pgm.process_group_manager.pp_rank])
        layers_per_gpu = [list(range(start_layer, start_layer + num_layers_per_gpu[pgm.process_group_manager.pp_rank]))]
        return layers_per_gpu

    def distribute_layers_interleaved(self, num_layers: int, num_virtual_pipeline_stages: int) -> List[List[int]]:
        print(f"Pipeline Parallel: Interleaved 1F1B pipeline with {num_virtual_pipeline_stages} virtual pipeline stages")
        
        pp_rank = pgm.process_group_manager.pp_rank
        pp_world_size = pgm.process_group_manager.pp_world_size
        
        num_layers_per_virtual_stage = num_layers // (pp_world_size * num_virtual_pipeline_stages)
        if num_layers_per_virtual_stage == 0:
            raise ValueError(f"Too many virtual pipeline stages ({num_virtual_pipeline_stages}) "
                            f"for the number of layers ({num_layers}) and pipeline stages ({pp_world_size})")
        
        # Calculate extra layers that need to be distributed
        extra_layers = num_layers % (pp_world_size * num_virtual_pipeline_stages)
        
        layers_per_gpu = []
        
        # Distribute layers across virtual pipeline stages
        for virtual_stage in range(num_virtual_pipeline_stages):
            base_idx = virtual_stage * pp_world_size * num_layers_per_virtual_stage
            
            # Calculate start and end indices for this pipeline stage's chunk
            chunk_start = base_idx + (pp_rank * num_layers_per_virtual_stage)
            chunk_end = chunk_start + num_layers_per_virtual_stage
            
            # Add extra layer if needed
            if extra_layers > 0:
                current_stage = virtual_stage * pp_world_size + pp_rank
                if current_stage < extra_layers:
                    chunk_end += 1
            
            # Add layers for this chunk to the current virtual stage
            if chunk_end <= num_layers:
                layers_in_virtual_stage = list(range(chunk_start, chunk_end))
                layers_per_gpu.append(layers_in_virtual_stage)
                        
        if not layers_per_gpu:
            raise ValueError(f"No layers assigned to pipeline stage {pp_rank}")
        
        print(f"Pipeline Parallel: Pipeline stage {pp_rank} assigned layers by virtual stage: {layers_per_gpu}")
        return layers_per_gpu
        
    def forward(self, input_ids, position_ids, hidden_states):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embedding(x)
        for layer in self.decoder_layers.values():
            x = layer(x, position_ids=position_ids)
        x = self.final_norm(x)
        return self.final_proj(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        # torch.autograd.backward will automatically accumulates gradients in the leaves (cf: https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """
    Executes a training step using Activation Forward - Activation Backward (AFAB) pipeline parallelism.
    Implements separate forward and backward passes to optimize memory usage.
    """
    logging_loss: torch.float32 = 0.0
    input_tensors, output_tensors = [], []
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1

    # === All Forward Pass Phase ===
    for _ in range(data_loader.grad_acc_steps):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # === All Backward Pass Phase ===
    for ith_microbatch in range(data_loader.grad_acc_steps):
        if requires_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss

def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):    
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1
    
    def _forward_step(input_tensor):
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps
        return output_tensor

    # === Warmup forward passes ===
    for _ in range(num_warmup_microbatches):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
    
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    # === 1F1B steady state ===
    for ith_microbatch in range(num_microbatches_remaining):
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # Trigger gradient sync on the last microbatch but only when last rank (the one that has num_warmup_microbatches = 0) has finished computing its backward pass.
        if num_warmup_microbatches == 0 and is_last_iteration:
            model.require_backward_grad_sync = True

        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        
        if is_last_iteration:
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    # === Cooldown backward passes ===
    for ith_warmup_microbatches in range(num_warmup_microbatches):
        if requires_grad_sync:
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = (ith_warmup_microbatches == num_warmup_microbatches - 1)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss

def train_step_pipeline_1f1b_interleaved(model, data_loader, tensor_shapes, device, dtype):    
    print("train_step_pipeline_1f1b_interleaved")
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    requires_grad_sync = pgm.process_group_manager.dp_world_size > 1
    
    def _forward_step(input_tensor):
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps
        return output_tensor

    # === Warmup forward passes ===
    for _ in range(num_warmup_microbatches):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
    
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    # === 1F1B steady state ===
    for ith_microbatch in range(num_microbatches_remaining):
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # Trigger gradient sync on the last microbatch but only when last rank (the one that has num_warmup_microbatches = 0) has finished computing its backward pass.
        if num_warmup_microbatches == 0 and is_last_iteration:
            model.require_backward_grad_sync = True

        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        
        if is_last_iteration:
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    # === Cooldown backward passes ===
    for ith_warmup_microbatches in range(num_warmup_microbatches):
        if requires_grad_sync:
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = (ith_warmup_microbatches == num_warmup_microbatches - 1)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss
### end Pipeline Parallel