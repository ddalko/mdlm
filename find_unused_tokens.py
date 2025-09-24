import json
from collections import Counter

import torch
import transformers


def analyze_token_usage():
    """데이터셋에서 실제로 사용되는 토큰들 분석"""
    print("=== 토큰 사용량 분석 ===")
    
    # GPT2 토크나이저 로드
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # 실제 데이터 로드
    try:
        with open('/workspace/mdlm/chat_templated_jsonschema_max1024.json', 'r') as f:
            data = json.load(f)
        print(f"데이터 개수: {len(data)}")
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return
    
    # 모든 토큰 사용량 카운트
    token_counter = Counter()
    
    print("토큰 사용량 계산 중...")
    for i, item in enumerate(data[:1000]):  # 처음 1000개만 분석 (시간 단축)
        if i % 100 == 0:
            print(f"  진행률: {i}/1000")
        
        # 프롬프트와 텍스트 토큰화
        prompt = item.get('prompt', '')
        text = item.get('text', prompt)
        
        # 토큰화
        tokens = tokenizer.encode(text, add_special_tokens=False)
        for token_id in tokens:
            token_counter[token_id] += 1
    
    print(f"분석된 총 토큰 개수: {sum(token_counter.values())}")
    print(f"사용된 고유 토큰 개수: {len(token_counter)}")
    
    # 사용되지 않는 토큰들 찾기
    all_token_ids = set(range(tokenizer.vocab_size))
    used_token_ids = set(token_counter.keys())
    unused_token_ids = all_token_ids - used_token_ids
    
    print(f"사용되지 않는 토큰 개수: {len(unused_token_ids)}")
    
    # 사용되지 않는 토큰 중 일부 출력
    unused_list = sorted(list(unused_token_ids))[:20]
    print(f"사용되지 않는 토큰 예시 (처음 20개):")
    for token_id in unused_list:
        try:
            token_str = tokenizer.decode([token_id])
            print(f"  토큰 ID {token_id}: '{token_str}'")
        except:
            print(f"  토큰 ID {token_id}: <디코딩 실패>")
    
    # 가장 적게 사용되는 토큰들
    least_used = token_counter.most_common()[-20:]
    print(f"\n가장 적게 사용되는 토큰들:")
    for token_id, count in least_used:
        try:
            token_str = tokenizer.decode([token_id])
            print(f"  토큰 ID {token_id} (사용 {count}회): '{token_str}'")
        except:
            print(f"  토큰 ID {token_id} (사용 {count}회): <디코딩 실패>")
    
    # 권장 토큰 선택
    if unused_token_ids:
        recommended_token = min(unused_token_ids)  # 가장 작은 ID 선택
        try:
            token_str = tokenizer.decode([recommended_token])
            print(f"\n권장 <|endofjson|> 대체 토큰:")
            print(f"  토큰 ID: {recommended_token}")
            print(f"  토큰 문자열: '{token_str}'")
            print(f"  사용 횟수: 0 (미사용)")
        except:
            print(f"\n권장 토큰 ID {recommended_token}는 디코딩할 수 없습니다.")
    else:
        # 가장 적게 사용되는 토큰 권장
        if least_used:
            recommended_token, usage_count = least_used[0]
            try:
                token_str = tokenizer.decode([recommended_token])
                print(f"\n권장 <|endofjson|> 대체 토큰 (가장 적게 사용됨):")
                print(f"  토큰 ID: {recommended_token}")
                print(f"  토큰 문자열: '{token_str}'")
                print(f"  사용 횟수: {usage_count}")
            except:
                print(f"\n권장 토큰 ID {recommended_token}는 디코딩할 수 없습니다.")
    
    return unused_token_ids, token_counter

def check_special_tokens():
    """특수 토큰들 확인"""
    print("\n=== 특수 토큰 확인 ===")
    
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    
    special_tokens = {
        'bos_token': tokenizer.bos_token,
        'eos_token': tokenizer.eos_token,
        'unk_token': tokenizer.unk_token,
        'sep_token': tokenizer.sep_token,
        'pad_token': tokenizer.pad_token,
        'cls_token': tokenizer.cls_token,
        'mask_token': tokenizer.mask_token,
    }
    
    for name, token in special_tokens.items():
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{name}: '{token}' (ID: {token_id})")
        else:
            print(f"{name}: None")
    
    # UNK 토큰이 있다면 그것을 사용할 수 있는지 확인
    if tokenizer.unk_token is not None:
        unk_id = tokenizer.unk_token_id
        print(f"\nUNK 토큰을 <|endofjson|> 대신 사용 가능:")
        print(f"  토큰 ID: {unk_id}")
        print(f"  토큰 문자열: '{tokenizer.unk_token}'")

if __name__ == "__main__":
    unused_tokens, token_counter = analyze_token_usage()
    check_special_tokens()