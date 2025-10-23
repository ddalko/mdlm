#!/usr/bin/env python3
"""
Visualize prompt_mask and json_structure_mask from dataloader output using Streamlit.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from dataloader import get_dataset

def _split_schemabench_dataset(train_set, valid_seed=None):
  """
  Split schemabench dataset into train and validation sets using configured ratio.
  
  Args:
    train_set: The full training dataset to split
    config: Configuration object containing train_valid_ratio
    valid_seed: Seed for reproducible splitting
    
  Returns:
    tuple: (train_subset, valid_subset)
  """
  split_ratio = 0.9  # Default 0.9 -> train, 0.1 -> valid
  split_seed = valid_seed if valid_seed is not None else 2
  
  # Calculate split sizes
  total_size = len(train_set)
  valid_portion = max(1, int(round(total_size * (1.0 - split_ratio))))
  train_portion = total_size - valid_portion
  
  # Create reproducible split
  generator = torch.Generator().manual_seed(split_seed)
  train_subset, valid_subset = torch.utils.data.random_split(
    train_set, [train_portion, valid_portion], generator=generator)
  
  return train_subset, valid_subset


def process_sample_data(
    tokens: List[int],
    token_strs: List[str],
    prompt_mask: torch.Tensor,
    json_structure_mask: torch.Tensor,
    tokenizer,
    sample_id: int,
    endofvalue_token_id: int = 102
) -> Dict[str, Any]:
    """
    Process a single sample to extract token information and statistics.
    
    Args:
        tokens: Token IDs
        token_strs: Token strings
        prompt_mask: Boolean tensor indicating prompt tokens
        json_structure_mask: Boolean tensor indicating structure tokens
        tokenizer: Tokenizer used
        sample_id: Sample ID
        endofvalue_token_id: Token ID for endofvalue marker (default: 102)
        
    Returns:
        Dictionary with processed data and statistics
    """
    # Convert tensors to lists if needed
    if isinstance(prompt_mask, torch.Tensor):
        prompt_mask = prompt_mask.tolist()
    if isinstance(json_structure_mask, torch.Tensor):
        json_structure_mask = json_structure_mask.tolist()
    
    # Filter out padding tokens
    valid_indices = [i for i, tok in enumerate(tokens) if tok != tokenizer.pad_token_id]
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_token_strs = [token_strs[i] for i in valid_indices]
    valid_prompt_mask = [prompt_mask[i] for i in valid_indices]
    valid_structure_mask = [json_structure_mask[i] for i in valid_indices]
    
    # Process token strings and check for endofvalue tokens
    processed_tokens = []
    for tok_id, tok_str in zip(valid_tokens, valid_token_strs):
        if tok_id == endofvalue_token_id:
            # Special display for endofvalue token
            tok_display = '|end_of_value|'
        else:
            # Replace special token symbols
            tok_display = tok_str.replace('Ġ', ' ')  # GPT-2 space marker (using visible space)
        processed_tokens.append(tok_display)
    
    # Determine token types
    token_types = []
    for tok_id, is_prompt, is_structure in zip(valid_tokens, valid_prompt_mask, valid_structure_mask):
        if tok_id == endofvalue_token_id:
            token_types.append('endofvalue')
        elif is_prompt:
            token_types.append('prompt')
        elif is_structure:
            token_types.append('structure')
        else:
            token_types.append('value')
    
    # Calculate statistics
    num_tokens = len(valid_tokens)
    num_prompt = sum(valid_prompt_mask)
    num_endofvalue = sum(1 for tok_id in valid_tokens if tok_id == endofvalue_token_id)
    num_response = num_tokens - num_prompt
    num_structure_in_response = sum(
        1 for i, is_struct in enumerate(valid_structure_mask) 
        if is_struct and not valid_prompt_mask[i]
    )
    num_value_in_response = sum(
        1 for i, is_struct in enumerate(valid_structure_mask) 
        if not is_struct and not valid_prompt_mask[i] and valid_tokens[i] != endofvalue_token_id
    )
    
    return {
        'id': sample_id,
        'tokens': valid_tokens,
        'token_strs': processed_tokens,
        'token_types': token_types,
        'prompt_mask': valid_prompt_mask,
        'structure_mask': valid_structure_mask,
        'num_tokens': num_tokens,
        'num_prompt': num_prompt,
        'num_endofvalue': num_endofvalue,
        'num_response': num_response,
        'num_structure_in_response': num_structure_in_response,
        'num_value_in_response': num_value_in_response,
    }


def load_and_process_dataset(
    dataset,
    tokenizer,
    max_samples: int = 50
) -> List[Dict[str, Any]]:
    """
    Load and process dataset samples.
    
    Args:
        dataset: Dataset with input_ids, prompt_mask, json_structure_mask
        tokenizer: Tokenizer used
        max_samples: Maximum number of samples to process
        
    Returns:
        List of processed sample dictionaries
    """
    samples = []
    for i in range(min(len(dataset), max_samples)):
        item = dataset[i]
        
        # Extract data
        tokens = item['input_ids'].tolist() if isinstance(item['input_ids'], torch.Tensor) else item['input_ids']
        prompt_mask = item.get('prompt_mask', torch.zeros(len(tokens), dtype=torch.bool))
        json_structure_mask = item.get('json_structure_mask', torch.zeros(len(tokens), dtype=torch.bool))
        
        # Get token strings
        token_strs = tokenizer.convert_ids_to_tokens(tokens)
        
        # Process sample
        sample_data = process_sample_data(
            tokens,
            token_strs,
            prompt_mask,
            json_structure_mask,
            tokenizer,
            i
        )
        
        samples.append(sample_data)
    
    return samples


def display_token_with_color(token_str: str, token_type: str):
    """Display a single token with appropriate color based on its type."""
    if token_type == 'prompt':
        st.markdown(
            f'<span style="background-color: #2196F3; color: white; padding: 2px 4px; '
            f'border-radius: 3px; font-weight: 600; font-family: monospace;">{token_str}</span>',
            unsafe_allow_html=True
        )
    elif token_type == 'structure':
        st.markdown(
            f'<span style="background-color: #FFF9C4; color: #F57F17; padding: 2px 4px; '
            f'border-radius: 3px; font-weight: 600; font-family: monospace;">{token_str}</span>',
            unsafe_allow_html=True
        )
    elif token_type == 'endofvalue':
        st.markdown(
            f'<span style="background-color: #9C27B0; color: white; padding: 2px 4px; '
            f'border-radius: 3px; font-weight: 600; font-family: monospace;">{token_str}</span>',
            unsafe_allow_html=True
        )
    else:  # value
        st.markdown(
            f'<span style="background-color: #ffebee; color: #d32f2f; padding: 2px 4px; '
            f'border-radius: 3px; font-weight: 600; font-family: monospace;">{token_str}</span>',
            unsafe_allow_html=True
        )


def display_sample(sample: Dict[str, Any]):
    """Display a single sample with Streamlit components."""
    sid = sample['id']
    nt = sample['num_tokens']
    np_tok = sample['num_prompt']
    nr = sample['num_response']
    ns = sample['num_structure_in_response']
    nv = sample['num_value_in_response']
    neov = sample.get('num_endofvalue', 0)
    sp = (ns / nr * 100) if nr > 0 else 0
    vp = (nv / nr * 100) if nr > 0 else 0
    eovp = (neov / nr * 100) if nr > 0 else 0
    
    with st.expander(
        f"**Sample #{sid}** | Tokens: {nt} (P:{np_tok} + R:{nr}) | Response: S:{sp:.1f}% / V:{vp:.1f}% / EOV:{eovp:.1f}%",
        expanded=False
    ):
        # Statistics
        st.markdown("### 📊 Token Statistics")
        cols = st.columns(5)
        with cols[0]:
            st.metric("Prompt Tokens", np_tok)
        with cols[1]:
            st.metric("Response Tokens", nr)
        with cols[2]:
            st.metric("Structure (Response)", f"{ns} ({sp:.1f}%)")
        with cols[3]:
            st.metric("Value (Response)", f"{nv} ({vp:.1f}%)")
        with cols[4]:
            st.metric("EndOfValue", f"{neov} ({eovp:.1f}%)")
        
        st.markdown("---")
        
        # Token visualization
        st.markdown("### 🔤 Token Visualization")
        
        # Group consecutive endofvalue tokens
        grouped_tokens = []
        grouped_types = []
        i = 0
        while i < len(sample['token_strs']):
            tok_str = sample['token_strs'][i]
            tok_type = sample['token_types'][i]
            
            if tok_type == 'endofvalue':
                # Count consecutive endofvalue tokens
                count = 1
                while i + count < len(sample['token_types']) and sample['token_types'][i + count] == 'endofvalue':
                    count += 1
                
                # Add grouped representation
                if count > 1:
                    grouped_tokens.append(f"|end_of_value| * {count}")
                else:
                    grouped_tokens.append(tok_str)
                grouped_types.append('endofvalue')
                i += count
            else:
                grouped_tokens.append(tok_str)
                grouped_types.append(tok_type)
                i += 1
        
        # Create HTML for all tokens
        html_parts = []
        for tok_str, tok_type in zip(grouped_tokens, grouped_types):
            # Escape HTML special characters
            tok_str_escaped = tok_str.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            
            if tok_type == 'prompt':
                html_parts.append(
                    f'<span style="background-color: #2196F3; color: white; padding: 2px 4px; '
                    f'border-radius: 3px; font-weight: 600; margin-right: 3px; '
                    f'font-family: monospace; white-space: pre;">{tok_str_escaped}</span>'
                )
            elif tok_type == 'structure':
                html_parts.append(
                    f'<span style="background-color: #FFF9C4; color: #F57F17; padding: 2px 4px; '
                    f'border-radius: 3px; font-weight: 600; margin-right: 3px; '
                    f'font-family: monospace; white-space: pre;">{tok_str_escaped}</span>'
                )
            elif tok_type == 'endofvalue':
                html_parts.append(
                    f'<span style="background-color: #9C27B0; color: white; padding: 2px 4px; '
                    f'border-radius: 3px; font-weight: 600; margin-right: 3px; '
                    f'font-family: monospace; white-space: pre;">{tok_str_escaped}</span>'
                )
            else:  # value
                html_parts.append(
                    f'<span style="background-color: #ffebee; color: #d32f2f; padding: 2px 4px; '
                    f'border-radius: 3px; font-weight: 600; margin-right: 3px; '
                    f'font-family: monospace; white-space: pre;">{tok_str_escaped}</span>'
                )
        
        html_content = ''.join(html_parts)
        st.markdown(
            f'<div style="line-height: 2.0; padding: 15px; background-color: #fafafa; '
            f'border-radius: 5px; border: 1px solid #e0e0e0;">{html_content}</div>',
            unsafe_allow_html=True
        )


def main_streamlit():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Dataloader Mask Visualization",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Dataloader Mask Visualization")
    
    # Legend
    st.markdown("### Legend")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<span style="background-color: #2196F3; color: white; padding: 4px 8px; '
            'border-radius: 3px; font-weight: 600;">Prompt</span> '
            '<span style="margin-left: 10px;">Blue: Prompt tokens</span>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<span style="background-color: #FFF9C4; color: #F57F17; padding: 4px 8px; '
            'border-radius: 3px; font-weight: 600;">Structure</span> '
            '<span style="margin-left: 10px;">Yellow: JSON structure</span>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<span style="background-color: #ffebee; color: #d32f2f; padding: 4px 8px; '
            'border-radius: 3px; font-weight: 600;">Value</span> '
            '<span style="margin-left: 10px;">Red: JSON values</span>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            '<span style="background-color: #9C27B0; color: white; padding: 4px 8px; '
            'border-radius: 3px; font-weight: 600;">&lt;|endofvalue|&gt;</span> '
            '<span style="margin-left: 10px;">Purple: End of value marker</span>',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Load data
    if 'samples' not in st.session_state:
        with st.spinner("Loading dataset and processing samples..."):
            tokenizer_name = "gpt2"
            data_file = "chat_templated_jsonschema_max1024_ws_1.json"
            block_size = 1024
            cache_dir = ".cache_viz_streamlit"
            use_endofvalue_token = True  # Enable endofvalue token insertion
            
            # Remove cache directory if it exists
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Set special tokens
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.cls_token if tokenizer.cls_token else '[BOS]'
            if tokenizer.eos_token is None:
                tokenizer.eos_token = tokenizer.sep_token if tokenizer.sep_token else '[EOS]'
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            st.info(f"Loading dataset with endofvalue_token={'enabled' if use_endofvalue_token else 'disabled'}...")
            
            dataset = get_dataset(
                'schemabench',
                tokenizer,
                mode='train',
                wrap=False,
                cache_dir=cache_dir,
                block_size=block_size,
                data_files={'train': data_file},
                use_endofvalue_token=use_endofvalue_token
            )
            
            train_set, valid_set = _split_schemabench_dataset(dataset)
            st.session_state.samples = load_and_process_dataset(valid_set, tokenizer, max_samples=50)
            
            # Clean up cache
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
    
    samples = st.session_state.samples
    
    # Summary statistics
    if samples:
        total_samples = len(samples)
        avg_prompt_tokens = sum(s['num_prompt'] for s in samples) / total_samples
        avg_response_tokens = sum(s['num_response'] for s in samples) / total_samples
        avg_endofvalue = sum(s.get('num_endofvalue', 0) for s in samples) / total_samples
        avg_structure_pct = sum(
            s['num_structure_in_response'] / s['num_response'] * 100 
            if s['num_response'] > 0 else 0 
            for s in samples
        ) / total_samples
        avg_value_pct = sum(
            s['num_value_in_response'] / s['num_response'] * 100 
            if s['num_response'] > 0 else 0 
            for s in samples
        ) / total_samples
        avg_eov_pct = sum(
            s.get('num_endofvalue', 0) / s['num_response'] * 100
            if s['num_response'] > 0 else 0
            for s in samples
        ) / total_samples
        total_tokens = sum(s['num_tokens'] for s in samples)
        
        st.markdown("### 📊 Overall Statistics")
        cols = st.columns(7)
        with cols[0]:
            st.metric("Total Samples", total_samples)
        with cols[1]:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with cols[2]:
            st.metric("Avg Prompt", f"{avg_prompt_tokens:.1f}")
        with cols[3]:
            st.metric("Avg Response", f"{avg_response_tokens:.1f}")
        with cols[4]:
            st.metric("Avg Structure %", f"{avg_structure_pct:.1f}%")
        with cols[5]:
            st.metric("Avg Value %", f"{avg_value_pct:.1f}%")
        with cols[6]:
            st.metric("Avg EndOfValue", f"{avg_endofvalue:.1f} ({avg_eov_pct:.1f}%)")
        
        st.markdown("---")
        
        # Display samples
        st.markdown("### 📝 Samples")
        for sample in samples:
            display_sample(sample)


def main():
    """Entry point - runs Streamlit app."""
    main_streamlit()


if __name__ == "__main__":
    main()
