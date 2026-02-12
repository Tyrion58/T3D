"""
DDO SDAR Training Script - Full Parameter Fine-tuning
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
import wandb
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from train.prepare import collect_training_data, make_basic_block_attention, process_pad
from train.utils import get_config, flatten_omega_conf

try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_models_full_finetune(config):
    """
    Full parameter fine-tuning version.
    Returns: (model, ref_model, tokenizer)
    - model: the trainable model
    - ref_model: frozen reference model (separate copy)
    """
    pretrained_model = config.model.pretrained_model
    
    # Load the trainable model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    
    # Load a separate frozen reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    # Freeze all parameters in reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_trainable:,}")

    return model, ref_model, tokenizer


def copy_model_weights(src_model, dst_model):
    """
    Copy all weights from src_model to dst_model for multi-round refinement.
    This is used in self-play to update the reference model after each round.
    """
    with torch.no_grad():
        src_state = src_model.state_dict()
        dst_model.load_state_dict(src_state)
    print("Copied model weights from training model to reference model")


class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, step_map_list):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.step_map_list = step_map_list
    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.step_map_list[idx]
        )

def simple_collate(batch):
    idx, extended_input_ids, p_mask, tok_idx_ext, labels, step_map_list = zip(*batch)
    
    step_map_tensors = []
    for sm in step_map_list:
        if isinstance(sm, (list, tuple)):
            step_map_tensors.append(torch.tensor(sm, dtype=torch.long))
        elif isinstance(sm, torch.Tensor):
            step_map_tensors.append(sm)
        else:
            step_map_tensors.append(torch.tensor(sm, dtype=torch.long))

    B = len(labels)
    if B > 0:
        max_len = max(sm.shape[0] if isinstance(sm, torch.Tensor) else len(sm) for sm in step_map_tensors)
        
        step_map_padded = []
        for sm in step_map_tensors:
            if sm.shape[0] < max_len:
                padding = torch.full((max_len - sm.shape[0],), -1, dtype=sm.dtype)
                sm = torch.cat([sm, padding], dim=0)
            step_map_padded.append(sm)
        
        step_map_stacked = torch.stack(step_map_padded)
    else:
        step_map_stacked = torch.empty((0, 0), dtype=torch.long)
    
    return {
        "ids": torch.tensor(idx),
        "extended_input_ids": torch.stack(extended_input_ids),
        "p_mask": torch.stack(p_mask),
        "tok_idx_ext": torch.stack(tok_idx_ext),
        "labels": torch.stack(labels),
        "step_map_list": step_map_stacked
    }


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    # Convert to float64 for precision
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise    


def compute_logp(model, ref_model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, basic_block_attention, pad_id, reference_logits=False):
    B, L = p_mask.shape
    L0    = start_pos
    L1    = L - L0
    device = extended_input_ids.device
    attention_mask = basic_block_attention.clone()
    attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
    attention_mask = process_pad(attention_mask, extended_input_ids, L0, L1, start_pos, pad_id)
    
    if reference_logits:
        # Use the frozen reference model
        logits = ref_model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
    else:
        # Use the trainable model
        logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
    
    logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1:, :]], dim=1)  # (B, L0+L1, V)
    log_probs = F.log_softmax(logits, dim=-1)
    logp_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    logp = (logp_tok * p_mask).sum(dim=1) / p_mask.sum(dim=1)  # (B,)
    return logp


def compute_logp_fake(model, ref_model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, basic_block_attention, pad_id, temperature, reference_logits=False, x_theta_fake=None):
    B, L = p_mask.shape
    L0    = start_pos
    L1    = L - L0

    device = extended_input_ids.device
    attention_mask = basic_block_attention.clone()
    attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
    attention_mask = process_pad(attention_mask, extended_input_ids, L0, L1, start_pos, pad_id)
    
    if reference_logits:
        # Use the frozen reference model
        logits = ref_model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
    else:
        # Use the trainable model
        logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits

    logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1:, :]], dim=1)  # (B, L0+L1, V)
    if x_theta_fake is None:
        if temperature == 0.0:
            x_theta = torch.argmax(logits, dim=-1)
        else:
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x_theta = torch.argmax(logits_with_noise, dim=-1)
    else:
        x_theta = x_theta_fake
    
    log_probs = F.log_softmax(logits, dim=-1)
    logp_tok = log_probs.gather(dim=-1, index=x_theta.unsqueeze(-1)).squeeze(-1)  # (B, T)
    logp = (logp_tok * p_mask).sum(dim=1) / p_mask.sum(dim=1)  # (B,)
    return logp, logits


def compute_loss(config, target_logp_real, target_logp_fake, reference_logp_real, reference_logp_fake):
    alpha = config.training.alpha
    beta = config.training.beta

    loss_real = -F.logsigmoid(beta * (target_logp_real - reference_logp_real))
    sigmoid_val = torch.sigmoid(beta * (target_logp_fake - reference_logp_fake))
    loss_fake = -alpha * torch.log(1 - sigmoid_val)

    loss = loss_real.mean() + loss_fake.mean()

    metrics = {
        'loss': loss.item(),
        'loss_real': loss_real.mean().item(),
        'loss_fake': loss_fake.mean().item(),
        'logp_target_real': target_logp_real.mean().item(),
        'logp_ref_real': reference_logp_real.mean().item(),
        'logp_target_fake': target_logp_fake.mean().item(),
        'logp_ref_fake': reference_logp_fake.mean().item(),
    }
    return loss, metrics


def save_checkpoint(model, tokenizer, config, accelerator, name, project_timestamp_dir):
    # Use the timestamp-based directory structure passed from main
    output_dir = project_timestamp_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_base / name,
            safe_serialization=True,
        )
        logger.info(f"Saved model to {save_base / name}")

        # Save tokenizer
        tokenizer.save_pretrained(str(save_base / name))

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / name / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        # Also save config.yaml to checkpoint directory for convenience
        OmegaConf.save(config, save_base / name / "config.yaml")

        logger.info(f"Saved model + tokenizer to {save_base / name}")


def reconstruct_xt_from_step_map(x0, step_map, block_size, k_steps, mask_id, pad_id, start_pos):
    """
    Reconstruct xt from x0 using step_map decode order.
    
    Args:
        x0: [B, L] clean tokens (full sequence including prefix)
        step_map: [B, L1] decode order for response tokens (lower = decoded earlier)
        block_size: size of each block
        k_steps: number of tokens per large step
        mask_id: mask token id
        pad_id: pad token id
        start_pos: start position of response (L0)
    
    Returns:
        List of xt tensors, one for each step level: [xt_step0, xt_step1, ..., xt_stepN]
        Each xt is [B, L] with masked tokens according to decode order
    """
    B, L = x0.shape
    L0 = start_pos
    L1 = L - L0
    device = x0.device
    
    # Extract response part
    x0_resp = x0[:, start_pos:]  # [B, L1]
    
    # Ensure step_map is 2D [B, L1_step] and matches L1 length
    if step_map.dim() == 1:
        # If 1D, assume it's for a single sample, expand to batch
        step_map = step_map.unsqueeze(0)  # [1, L1_step]
        if step_map.shape[0] != B:
            step_map = step_map.expand(B, -1)  # [B, L1_step]
    
    step_map_L1 = step_map.shape[1]
    if step_map_L1 != L1:
        # Trim or pad step_map to match L1
        if step_map_L1 > L1:
            # Trim step_map to L1
            step_map = step_map[:, :L1]  # [B, L1]
        else:
            # Pad step_map to L1 with -1 (invalid values)
            pad_len = L1 - step_map_L1
            step_map = torch.cat([
                step_map,
                torch.full((B, pad_len), -1, dtype=step_map.dtype, device=device)
            ], dim=1)  # [B, L1]
    
    # Calculate number of blocks
    NB = (L1 + block_size - 1) // block_size  # Number of blocks
    steps_per_block = (block_size + k_steps - 1) // k_steps  # Steps per block
    total_steps = NB * steps_per_block
    
    # Pad step_map and x0_resp to be divisible by block_size
    pad_len = NB * block_size - L1
    if pad_len > 0:
        step_map_pad = torch.cat([
            step_map,
            torch.full((B, pad_len), -1, dtype=step_map.dtype, device=device)
        ], dim=1)  # [B, NB*block_size]
        x0_resp_pad = torch.cat([
            x0_resp,
            torch.full((B, pad_len), pad_id, dtype=x0_resp.dtype, device=device)
        ], dim=1)  # [B, NB*block_size]
    else:
        step_map_pad = step_map
        x0_resp_pad = x0_resp
    
    # Reshape into blocks: [B, NB, block_size]
    step_map_blocks = step_map_pad.view(B, NB, block_size)
    x0_resp_blocks = x0_resp_pad.view(B, NB, block_size)
    
    # Store all step-level xt tensors
    xt_list = []
    
    # For each step level
    for step_level in range(steps_per_block):
        # For each block, determine which tokens should be decoded at this step level
        # Step level s should decode the first (s+1)*k_steps tokens (by decode order) in each block
        # Initialize mask: all tokens should be masked initially
        mask_blocks = torch.ones(B, NB, block_size, dtype=torch.bool, device=device)
        
        # For each block, find tokens that should be decoded at this step level
        for bi in range(NB):
            block_step_map = step_map_blocks[:, bi, :]  # [B, block_size]
            
            # For each sample in batch, find which tokens to decode in this block
            for b in range(B):
                block_sm = block_step_map[b]  # [block_size]
                
                # Find valid (non-negative) step_map values
                valid_mask = block_sm >= 0
                valid_count = valid_mask.sum().item()
                
                if valid_count == 0:
                    # No valid tokens, keep all masked
                    continue
                
                # Number of tokens to decode at this step level
                num_to_decode = min((step_level + 1) * k_steps, valid_count)
                
                if num_to_decode > 0:
                    # Get valid step_map values and their indices
                    valid_step_map = block_sm[valid_mask]  # [valid_count]
                    valid_indices = torch.where(valid_mask)[0]  # [valid_count]
                    
                    # Sort by step_map to get decode order
                    sorted_indices = torch.argsort(valid_step_map)
                    # Get indices of tokens to decode
                    decode_indices = valid_indices[sorted_indices[:num_to_decode]]
                    
                    # Unmask these tokens
                    mask_blocks[b, bi, decode_indices] = False
                
                # Also keep tokens that are already decoded (step_map == -1) - these should remain unmasked
                already_decoded = block_sm == -1
                mask_blocks[b, bi, :] = mask_blocks[b, bi, :] & (~already_decoded)
        
        # Apply masking to response blocks
        xt_resp_blocks = x0_resp_blocks.clone()
        xt_resp_blocks[mask_blocks] = mask_id
        
        # Handle pad tokens: don't mask pad positions
        pad_mask = x0_resp_blocks == pad_id
        xt_resp_blocks[pad_mask] = pad_id
        
        # Flatten back to [B, NB*block_size]
        xt_resp_pad = xt_resp_blocks.view(B, -1)
        
        # Remove padding to get [B, L1]
        xt_resp = xt_resp_pad[:, :L1]
        
        # Reconstruct full sequence: prefix + masked response
        xt = torch.cat([x0[:, :start_pos], xt_resp], dim=1)  # [B, L]
        
        xt_list.append(xt)
    
    return xt_list


def generate_xt_from_step_map(x0, step_map, block_size, k_steps, mask_id, pad_id, start_pos):
    """
    Generate true xt from ground truth x0 using step_map decode order.
    Same logic as reconstruct_xt_from_step_map but uses ground truth x0.
    
    Args:
        x0: [B, L] ground truth clean tokens
        step_map: [B, L1] decode order for response tokens
        block_size: size of each block
        k_steps: number of tokens per large step
        mask_id: mask token id
        pad_id: pad token id
        start_pos: start position of response (L0)
    
    Returns:
        List of xt tensors, one for each step level: [xt_step0, xt_step1, ..., xt_stepN]
    """
    # Same implementation as reconstruct_xt_from_step_map
    return reconstruct_xt_from_step_map(x0, step_map, block_size, k_steps, mask_id, pad_id, start_pos)


def compute_path_loss_from_step_map(logits_x0, x0_true, step_map, block_size, k_steps, mask_id, pad_id, start_pos):
    """
    Compute path loss between predicted xt and true xt using step_map decode order.
    
    Args:
        logits_x0: [B, L, V] logits from target model
        x0_true: [B, L] ground truth clean tokens
        step_map: [B, L1] decode order for response tokens
        block_size: size of each block
        k_steps: number of tokens per large step
        mask_id: mask token id
        pad_id: pad token id
        start_pos: start position of response (L0)
    
    Returns:
        Scalar loss value (averaged across all step levels)
    """
    device = logits_x0.device
    B, L, V = logits_x0.shape
    L0 = start_pos
    L1 = L - L0
    
    # Get predicted x0 from logits
    x0_pred = torch.argmax(logits_x0, dim=-1)  # [B, L]
    
    # Reconstruct xt_predicted for all step levels
    xt_pred_list = reconstruct_xt_from_step_map(x0_pred, step_map, block_size, k_steps, mask_id, pad_id, start_pos)
    
    # Generate xt_true for all step levels
    xt_true_list = generate_xt_from_step_map(x0_true, step_map, block_size, k_steps, mask_id, pad_id, start_pos)
    # import ipdb; ipdb.set_trace()
    # Compute loss for each step level
    step_losses = []
    
    for xt_pred, xt_true in zip(xt_pred_list, xt_true_list):
        # Only compute loss on response part (start_pos onwards)
        # xt_pred_resp = xt_pred[:, start_pos:]  # [B, L1]
        xt_true_resp = xt_true[:, start_pos:]  # [B, L1]
        x0_true_resp = x0_true[:, start_pos:]  # [B, L1]
        
        # Compute loss on non-masked positions (where tokens are decoded, not masked)
        # Also exclude pad tokens
        # Non-masked positions: xt_true_resp != mask_id and x0_true_resp != pad_id
        non_masked_positions = (xt_true_resp != mask_id) & (x0_true_resp != pad_id)
        
        if not non_masked_positions.any():
            # No non-masked positions in this step level, skip
            continue
        
        # Get logits for response part
        logits_resp = logits_x0[:, start_pos:, :]  # [B, L1, V]
        
        # Compute cross-entropy loss on non-masked positions
        # Flatten for easier indexing
        non_masked_positions_flat = non_masked_positions.view(-1)  # [B*L1]
        logits_flat = logits_resp.view(-1, V)  # [B*L1, V]
        targets_flat = xt_true_resp.view(-1)  # [B*L1]
        
        # Get logits and targets for non-masked positions only
        logits_non_masked = logits_flat[non_masked_positions_flat]  # [N_non_masked, V]
        targets_non_masked = targets_flat[non_masked_positions_flat]  # [N_non_masked]
        
        if len(targets_non_masked) > 0:
            # Compute cross-entropy loss
            loss_step = F.cross_entropy(logits_non_masked, targets_non_masked, reduction='mean')
            step_losses.append(loss_step)
    
    # Average losses across all step levels
    if len(step_losses) > 0:
        total_loss = sum(step_losses) / len(step_losses)
    else:
        # No valid step levels, return zero loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return total_loss


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

    # Create timestamp-based directory structure
    timestamp = time.strftime("%Y%m%d_%H%M")
    project_timestamp_dir = Path("experiments") / pretrained_model.split("/")[-1] / config.dataset.data_type / project_name / timestamp
    project_timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(project_timestamp_dir / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(project_timestamp_dir, exist_ok=True)
        config_path = project_timestamp_dir / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)


    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer (Full Parameter Fine-tuning)")
    
    # Check if multi-round refinement is enabled
    multi_round = getattr(config.training, "multi_round", False)
    round_interval = getattr(config.training, "round_interval", 100)  # global steps per round
    
    if multi_round:
        logger.info(f"Multi-round refinement enabled with round_interval={round_interval} global steps")
    
    # Full parameter fine-tuning: get trainable model and frozen reference model
    model, ref_model, tokenizer = get_models_full_finetune(config)

    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    
    # calculate loss ourselves, needs logits, so avoid fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False
    if hasattr(ref_model, "config"):
        ref_model.config.fuse_cross_entropy = False

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)
    
    # Move reference model to device (no gradient checkpointing needed since it's frozen)
    ref_model = ref_model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)


    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # Optimize all trainable parameters (full fine-tuning)
    params = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "weight_decay": optimizer_config.weight_decay,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            params,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay if optimizer_config.weight_decay is not None else 0.0,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Print parameter information for monitoring
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info("PARAMETER MONITORING (Full Fine-tuning)")
        params_list = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        total_params = sum(p.numel() for _, p in params_list)
        logger.info(f"Total trainable parameters: {total_params:,}")


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    with open("./data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)

    prompt_list = []
    response_list = []
    step_map_list = []
    for x in dataset_load:
        prompt_list.append(x["question"])
        answer = x["answer"]
                # Handle case where answer is a list (extract first element) or a string
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        response_list.append(answer)
        step_map_list.append(x["step_map"][0])

    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))

    _, L = input_ids_lm.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num

    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()
    
    # Add vocab_size to config for collect_training_data
    vocab_size = tokenizer.vocab_size
    
    extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data(config, input_ids_lm, start_pos, pad_id, mask_id, vocab_size, post_num, step_map_list=step_map_list)

    step_map_list_expanded = step_map_list

    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, step_map_list_expanded)

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    # Use torch.Generator to set seed for DataLoader shuffle
    generator = None
    if hasattr(config.training, 'seed') and config.training.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(config.training.seed)

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0,
        shuffle=True,
        generator=generator,
    )


    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running SDAR training (Full Fine-tuning) *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num training data = {len(dataset_lm)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    
    first_epoch = 0
    global_step = 0
    end = time.time()

    from tqdm.auto import tqdm

    for epoch in range(first_epoch, num_train_epochs):

        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,    
            leave=True          
        )
        
        # Initialize loss accumulator for current accumulation window
        loss_accumulator = {
            "loss": 0.0,
            "loss_path_step_map": 0.0,
            "loss_ddo": 0.0,
            "loss_real": 0.0,
            "loss_fake": 0.0,
            "logp_target_real": 0.0,
            "logp_ref_real": 0.0,
            "logp_target_fake": 0.0,
            "logp_ref_fake": 0.0,
        }
        # Track how many times each loss was computed in the accumulation window
        loss_count = {
            "loss": 0,
            "loss_path_step_map": 0,
            "loss_ddo": 0,
            "loss_real": 0,
            "loss_fake": 0,
            "logp_target_real": 0,
            "logp_ref_real": 0,
            "logp_target_fake": 0,
            "logp_ref_fake": 0,
        }
        
        for step, batch in enumerate(progress_bar, start=1):

            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            
            # accumulate the gradient of the trainable model
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                with torch.no_grad():
                    # 3. Compute the reference model real logp
                    reference_logp_real = compute_logp(model, ref_model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, basic_block_attention, pad_id, reference_logits=True)
                    # 4. Compute the reference model fake logp
                    reference_logp_fake, logits_x0_fake = compute_logp_fake(model, ref_model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, basic_block_attention, pad_id, temperature=config.training.reference_temp, reference_logits=True, x_theta_fake=None)

                    if config.training.reference_temp == 0.0:
                        x_theta_fake = torch.argmax(logits_x0_fake, dim=-1)
                    else:
                        logits_with_noise = add_gumbel_noise(logits_x0_fake, temperature=config.training.reference_temp)
                        x_theta_fake = torch.argmax(logits_with_noise, dim=-1)

                # 1. Compute target model real logp
                target_logp_real = compute_logp(model, ref_model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, basic_block_attention, pad_id, reference_logits=False)

                # 2. Compute target model fake logp
                target_logp_fake, logits_x0_theta = compute_logp_fake(model, ref_model, extended_input_ids, p_mask, tok_idx_ext, labels, start_pos, basic_block_attention, pad_id, temperature=config.training.target_temp, reference_logits=False, x_theta_fake=x_theta_fake)


                # 5. Compute the loss
                loss, metrics = compute_loss(config, target_logp_real, target_logp_fake, reference_logp_real, reference_logp_fake)

                # Add step_map-based path loss
                step_map = batch["step_map_list"].to(accelerator.device)  # [B, L1]
                k_steps = getattr(config.training, "path_k_steps", 2)
                lambda_path = getattr(config.training, "lambda_path", 0.05)
                
                # Compute path loss from step_map
                if step_map.dtype != torch.long:
                    step_map = step_map.long()
                
                L_path_step_map = compute_path_loss_from_step_map(
                    logits_x0_theta, labels, step_map, 
                    config.training.block_size, k_steps, 
                    mask_id, pad_id, start_pos
                )
                
                # Add to total loss
                loss = loss + lambda_path * L_path_step_map

                loss_accumulator['loss'] += metrics['loss']
                loss_accumulator['loss_path_step_map'] += L_path_step_map.item()
                loss_accumulator['loss_ddo'] += loss.item()
                loss_accumulator['loss_real'] += metrics['loss_real']
                loss_accumulator['loss_fake'] += metrics['loss_fake']
                loss_accumulator['logp_target_real'] += metrics['logp_target_real']
                loss_accumulator['logp_ref_real'] += metrics['logp_ref_real']
                loss_accumulator['logp_target_fake'] += metrics['logp_target_fake']
                loss_accumulator['logp_ref_fake'] += metrics['logp_ref_fake']

                loss_count['loss'] += 1
                loss_count['loss_path_step_map'] += 1
                loss_count['loss_ddo'] += 1
                loss_count['loss_real'] += 1
                loss_count['loss_fake'] += 1
                loss_count['logp_target_real'] += 1
                loss_count['logp_ref_real'] += 1
                loss_count['logp_target_fake'] += 1
                loss_count['logp_ref_fake'] += 1
                
                # 6. Backward the loss
                accelerator.backward(loss)

                if config.experiment.gradient_clipping and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.experiment.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

            # Log and reset every gradient_accumulation_steps
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                global_step += 1
                
                # Calculate average losses over the accumulation window
                avg_losses = {}
                for key in loss_accumulator.keys():
                    if loss_count[key] > 0:
                        avg_losses[key] = loss_accumulator[key] / loss_count[key]
                    else:
                        avg_losses[key] = 0.0
                
                # Print losses (without detailed mask ratio statistics)
                if accelerator.is_main_process:
                    grad_norm = model.get_global_grad_norm() if hasattr(model, 'get_global_grad_norm') else None
                    current_round = ((global_step - 1) // round_interval) + 1 if multi_round else 0
                    total_rounds = max_train_steps // round_interval if multi_round else 0
                    print("=" * 80)
                    if multi_round:
                        print(f"Round {current_round}/{total_rounds} | Epoch {epoch+1}/{num_train_epochs} | Global Step {global_step} | Local Step {step}/{len(train_dataloader_lm)}")
                    else:
                        print(f"Epoch {epoch+1}/{num_train_epochs} | Global Step {global_step} | Local Step {step}/{len(train_dataloader_lm)}")
                    print("-" * 80)
                    print(f"Learning Rate:         {lr_scheduler.get_last_lr()[0]}")
                    print(f"Loss:                  {avg_losses['loss']:.4f}")
                    print(f"Loss Path Step Map:    {avg_losses['loss_path_step_map']:.4f}")
                    print(f"Loss DDO:              {avg_losses['loss_ddo']:.4f}")
                    print(f"Loss Real:             {avg_losses['loss_real']:.4f}")
                    print(f"Loss Fake:             {avg_losses['loss_fake']:.4f}")
                    print(f"Logp Target Real:      {avg_losses['logp_target_real']:.4f}")
                    print(f"Logp Ref Real:         {avg_losses['logp_ref_real']:.4f}")
                    print(f"Logp Target Fake:      {avg_losses['logp_target_fake']:.4f}")
                    print(f"Logp Ref Fake:         {avg_losses['logp_ref_fake']:.4f}")
                    print(f"Grad Norm:             {grad_norm.item() if grad_norm is not None else 0.0:.4f}")
                    print("=" * 80)
                
                # Log to wandb
                if accelerator.is_main_process:
                    grad_norm = model.get_global_grad_norm() if hasattr(model, 'get_global_grad_norm') else None
                    current_round = ((global_step - 1) // round_interval) + 1 if multi_round else 0
                    log_dict = {
                        "loss": avg_losses['loss'],
                        "loss_path_step_map": avg_losses['loss_path_step_map'],
                        "loss_ddo": avg_losses['loss_ddo'],
                        "loss_real": avg_losses['loss_real'],
                        "loss_fake": avg_losses['loss_fake'],
                        "logp_target_real": avg_losses['logp_target_real'],
                        "logp_ref_real": avg_losses['logp_ref_real'],
                        "logp_target_fake": avg_losses['logp_target_fake'],
                        "logp_ref_fake": avg_losses['logp_ref_fake'],
                        "grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    }
                    if multi_round:
                        log_dict["round"] = current_round
                    accelerator.log(log_dict, step=global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{avg_losses['loss']:.4f}",
                    'loss_real': f"{avg_losses['loss_real']:.4f}",
                    'loss_fake': f"{avg_losses['loss_fake']:.4f}",
                })
                
                # Reset accumulators for next window
                for key in loss_accumulator:
                    loss_accumulator[key] = 0.0
                    loss_count[key] = 0

                # Save checkpoint at specified global steps
                if global_step % config.experiment.get("save_every_steps", 128) == 0:
                    accelerator.wait_for_everyone()
                    logger.info(f"About to save checkpoint at global step {global_step}")
                    name = f"{config.model.optimized_name}-{epoch}-{global_step}"
                    save_checkpoint(model, tokenizer, config, accelerator, name, project_timestamp_dir)
                    logger.info(f"Saved checkpoint at global step {global_step}")
                
                # Multi-Round Refinement via Self-Play
                # At the end of each round (every round_interval global steps), update the reference model
                # by copying weights from the trained model to the reference model
                if multi_round and (global_step % round_interval == 0):
                    current_round = global_step // round_interval
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info("=" * 80)
                        logger.info(f"MULTI-ROUND REFINEMENT: Completing Round {current_round}")
                        logger.info(f"Updating reference model with trained model weights at global step {global_step}")
                        logger.info("=" * 80)
                    
                    # Copy weights from trained model to reference model
                    unwrapped_model = accelerator.unwrap_model(model)
                    copy_model_weights(unwrapped_model, ref_model)
                    
                    # Save checkpoint at round boundary
                    name = f"{config.model.optimized_name}-round{current_round}-step{global_step}"
                    save_checkpoint(model, tokenizer, config, accelerator, name, project_timestamp_dir)
                    
                    if accelerator.is_main_process:
                        logger.info(f"Round {current_round} completed. Reference model updated for Round {current_round + 1}")
                        logger.info("=" * 80)
    
    accelerator.wait_for_everyone()

    # Save final checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name, project_timestamp_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()

