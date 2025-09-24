#!/usr/bin/env python3
"""
Dataloader 테스트 스크립트
실제 schemabench 데이터셋을 로드해서 process_json_with_endofjson_token 함수를 테스트합니다.
"""

import json
import os
import sys

import transformers

from dataloader import (JSON_STRUCTURE_TOKEN_IDS, get_dataset,
                        get_json_structure_mask, get_tokenizer,
                        process_json_with_endofjson_token)


def test_tokenizer_setup():
    """토크나이저 설정 테스트"""
    print("=== Tokenizer Setup Test ===")
    
    # GPT2 토크나이저 설정 (실제 사용하는 것과 동일)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print(f"Tokenizer: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # 토큰 102 테스트
    token_102 = tokenizer.decode([102])
    print(f"Token ID 102 decodes to: '{token_102}' (repr: {repr(token_102)})")
    
    return tokenizer


def test_json_structure_mask():
    """JSON 구조 마스크 테스트"""
    print("\n=== JSON Structure Mask Test ===")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    
    # 테스트 JSON
    test_json = '{"id": 123, "name": "test", "active": true}'
    tokens = tokenizer.encode(test_json)
    
    print(f"Test JSON: {test_json}")
    print(f"Tokens: {tokens}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens]}")
    
    # 구조 마스크 생성
    structure_mask = get_json_structure_mask(tokens)
    structure_indices = [i for i, is_struct in enumerate(structure_mask) if is_struct]
    
    print(f"Structure mask: {structure_mask.tolist()}")
    print(f"Structure indices: {structure_indices}")
    print(f"Structure tokens: {[tokens[i] for i in structure_indices]}")
    print(f"Structure token strings: {[tokenizer.decode([tokens[i]]) for i in structure_indices]}")
    
    return structure_mask


def test_process_function_simple():
    """간단한 process_json_with_endofjson_token 함수 테스트"""
    print("\n=== Process Function Simple Test ===")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 테스트 데이터
    prompt = "Generate a JSON object:"
    response = '{"id": 123, "name": "test", "active": true}'
    text = prompt + " " + response
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Full text: {text}")
    
    # 함수 실행
    result = process_json_with_endofjson_token(text, prompt, tokenizer, 1024, 102)
    
    print(f"\nResult keys: {result.keys()}")
    print(f"Input IDs length: {len(result['input_ids'])}")
    print(f"Token 102 count: {result['input_ids'].count(102)}")
    
    # 패딩 제거한 실제 토큰들
    non_pad_tokens = [t for t in result['input_ids'] if t != tokenizer.pad_token_id]
    print(f"Non-pad tokens length: {len(non_pad_tokens)}")
    
    # 디코딩해서 확인
    decoded = tokenizer.decode(non_pad_tokens, skip_special_tokens=False)
    print(f"Decoded result: {decoded[:200]}...")
    
    return result


def test_real_dataset_sample():
    """실제 데이터셋 샘플 테스트"""
    print("\n=== Real Dataset Sample Test ===")
    
    # 실제 설정과 동일하게 설정
    class MockConfig:
        def __init__(self):
            self.data = MockDataConfig()
            self.model = MockModelConfig()
    
    class MockDataConfig:
        def __init__(self):
            self.train = 'schemabench'
            self.cache_dir = '/workspace/data'
            self.wrap = False
            self.tokenizer_name_or_path = 'gpt2'
            self.data_files = '/workspace/mdlm/chat_templated_jsonschema_max1024.json'
    
    class MockModelConfig:
        def __init__(self):
            self.length = 1024
    
    config = MockConfig()
    
    # 토크나이저 설정
    tokenizer = get_tokenizer(config)
    
    print(f"Tokenizer loaded: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    try:
        # 데이터셋 로드 시도
        dataset = get_dataset(
            dataset_name=config.data.train,
            tokenizer=tokenizer,
            mode='train',
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            data_files=config.data.data_files,
            use_endofjson_token=True,
            num_proc=1
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset length: {len(dataset)}")
        
        # 첫 번째 샘플 확인
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample keys: {sample.keys()}")
            
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                print(f"Input IDs length: {len(input_ids)}")
                print(f"Token 102 count: {input_ids.count(102) if hasattr(input_ids, 'count') else sum(1 for t in input_ids if t == 102)}")
                
                # 실제 토큰들 (패딩 제외)
                if hasattr(input_ids, 'tolist'):
                    input_ids = input_ids.tolist()
                non_pad_tokens = [t for t in input_ids if t != tokenizer.pad_token_id]
                print(f"Non-pad tokens length: {len(non_pad_tokens)}")
                
                # 디코딩
                decoded = tokenizer.decode(non_pad_tokens[:100], skip_special_tokens=False)  # 처음 100개만
                print(f"Decoded sample (first 100 tokens): {decoded}")
            
            # 마스크 정보 확인
            if 'prompt_mask' in sample:
                prompt_mask = sample['prompt_mask']
                if hasattr(prompt_mask, 'tolist'):
                    prompt_mask = prompt_mask.tolist()
                prompt_length = sum(prompt_mask)
                print(f"Prompt mask length: {len(prompt_mask)}")
                print(f"Actual prompt length: {prompt_length}")
            
            if 'json_structure_mask' in sample:
                json_structure_mask = sample['json_structure_mask']
                if hasattr(json_structure_mask, 'tolist'):
                    json_structure_mask = json_structure_mask.tolist()
                structure_count = sum(json_structure_mask)
                print(f"JSON structure mask length: {len(json_structure_mask)}")
                print(f"Structure token count: {structure_count}")
        
        return dataset
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print("Trying manual data loading...")
        
        # 수동으로 데이터 파일 읽기
        data_file = '/workspace/data/schemabench/train.jsonl'
        if os.path.exists(data_file):
            print(f"Found data file: {data_file}")
            with open(data_file, 'r') as f:
                lines = f.readlines()[:5]  # 처음 5줄만
                
            print(f"Data file has {len(lines)} lines (showing first 5)")
            
            for i, line in enumerate(lines):
                try:
                    data = json.loads(line.strip())
                    print(f"\nSample {i}:")
                    print(f"Keys: {data.keys()}")
                    
                    if 'text' in data and 'prompt' in data:
                        text = data['text']
                        prompt = data['prompt']
                        
                        print(f"Prompt length: {len(prompt)}")
                        print(f"Text length: {len(text)}")
                        print(f"Prompt: {prompt[:100]}...")
                        print(f"Text: {text[:100]}...")
                        
                        # process_json_with_endofjson_token 테스트
                        result = process_json_with_endofjson_token(text, prompt, tokenizer, 1024, 102)
                        print(f"Processed - Token 102 count: {result['input_ids'].count(102)}")
                        
                        break  # 첫 번째 성공한 샘플만 처리
                        
                except json.JSONDecodeError as je:
                    print(f"JSON decode error in line {i}: {je}")
                except Exception as pe:
                    print(f"Processing error in line {i}: {pe}")
        else:
            print(f"Data file not found: {data_file}")
            
        return None


def main():
    """메인 함수"""
    print("Dataloader 테스트 시작...")
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"Python 경로: {sys.path[:3]}")
    
    try:
        # 1. 토크나이저 테스트
        # tokenizer = test_tokenizer_setup()
        
        # 2. JSON 구조 마스크 테스트
        # test_json_structure_mask()
        
        # 3. 간단한 함수 테스트
        # test_process_function_simple()
        
        # 4. 실제 데이터셋 테스트
        dataset = test_real_dataset_sample()
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()