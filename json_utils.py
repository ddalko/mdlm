import re
import json
import jsonschema

from typing import List

class ParserError(Exception):
    pass


class ValidationError(Exception):
    pass


JSON_REGEX = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)

def safe_json_loads(txt: str):
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        end = min(e.pos + 20, len(txt))
        snippet = txt[:end].replace("\n", "\\n")
        raise ValueError(f"JSON parse failed at pos {e.pos}: {e.msg}. Snippet: {snippet}") from e

def extract_pred(seq: str):
    assistant_match = re.search(r"### Assistant:\s*(.*)$", seq, re.DOTALL | re.IGNORECASE)
    if not assistant_match:
        raise ValueError("Could not find '### Assistant:' in the text")
    
    return assistant_match.group(1).strip()

class CodeBlockJsonParser:
    """Parse JSON object contained in ```json\n \n```"""

    def loads(self, s: str):
        m = JSON_REGEX.search(s)
        if m is not None:
            return safe_json_loads(m.group(1).strip())
        return safe_json_loads(s)


def travese_and_convert(obj, schema):
    """Try to automatically convert boolean, number, integer, null."""
    if "type" in schema:
        t = schema["type"]

        if t == "object":
            newobj = {}
            if "properties" in schema:
                for key in schema["properties"]:
                    newobj[key] = travese_and_convert(obj[key], schema["properties"][key])
            if "patternProperties" in schema:
                # 최소 수정: 원래 코드와 동일한 접근(키 직접 매칭) 유지
                for key in schema["patternProperties"]:
                    newobj[key] = travese_and_convert(obj[key], schema["patternProperties"][key])
                # 개선안(패턴 매칭 사용) 예시:
                # import re
                # for pattern, subschema in schema["patternProperties"].items():
                #     for k, v in obj.items():
                #         if re.fullmatch(pattern, k):
                #             newobj[k] = travese_and_convert(v, subschema)

            if "maxProperties" in schema:
                raise NotImplementedError
            return newobj

        elif t == "array":
            arr = []
            if "items" in schema:
                items = schema["items"]
                if isinstance(items, list):
                    # multiple items mixed schema
                    for item in obj:
                        valid = False
                        for subschema in items:
                            try:
                                arr.append(travese_and_convert(item, subschema))
                                valid = True
                                break
                            except Exception:
                                pass
                        if not valid:
                            arr.append(item)
                else:
                    for item in obj:
                        arr.append(travese_and_convert(item, items))
            return arr

        elif t == "string":
            return str(obj)

        elif t == "number":
            try:
                return int(obj)
            except Exception:
                return float(obj)

        elif t == "integer":
            return int(obj)

        elif t == "boolean":
            if isinstance(obj, bool):
                return obj
            if isinstance(obj, str):
                low = obj.lower()
                if low == "true":
                    return True
                if low == "false":
                    return False
                raise ValueError(f"Invalid boolean value {obj}")
            return bool(obj)

        elif t == "null":
            if isinstance(obj, str):
                if obj.lower() == "null" or obj == "" or obj == "None":
                    return None
                raise ValueError(f"Invalid null value {obj}")
            return obj

        else:
            return obj

    else:
        if "allOf" in schema:
            for subschema in schema["allOf"]:
                obj = travese_and_convert(obj, subschema)
            return obj
        elif "anyOf" in schema:
            for subschema in schema["anyOf"]:
                try:
                    return travese_and_convert(obj, subschema)
                except Exception:
                    pass
            return obj
        elif "oneOf" in schema:
            for subschema in schema["oneOf"]:
                try:
                    return travese_and_convert(obj, subschema)
                except Exception:
                    pass
            return obj
        else:
            return obj



def validate_loads(s: str, schema: dict):
    """Validate the given string against the schema, return loaded object if valid."""
    obj = json.loads(s)

    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        # for bool, number, integer, None, we try to automatic convert them.
        obj = travese_and_convert(obj, schema)  # get json object recursively with automatic type conversion
        jsonschema.validate(obj, schema)

    return obj


def validate(pred: str, verify_schema: dict) -> bool:
    # first load and validate the pred
    try:
        codeblockjsonparser = CodeBlockJsonParser()
        loaded = codeblockjsonparser.loads(pred)
    except Exception as e:
        # raise ParserError("Failed to load the pred. " + str(e))
        raise ParserError(str(e))
    try:
        pred = validate_loads(pred, schema=verify_schema)
        score = True if pred is not None else False
        return score
    except Exception as e:
        # raise ValidationError("Failed to validate the pred aginst the schema. " + str(e))
        raise ValidationError(str(e))


def extract_schema(text: str) -> dict:
    """
    Extract JSON from the response part (after '### Assistant:')
    
    Args:
        text: Full text containing the assistant response
        
    Returns:
        dict: Parsed JSON from the response
    """
    # Find the assistant response part
    assistant_match = re.search(r"### Assistant:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if not assistant_match:
        raise ValueError("Could not find '### Assistant:' in the text")
    
    response_text = assistant_match.group(1).strip()
    
    # Try to parse as JSON directly
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to find JSON in code blocks
        codeblock_parser = CodeBlockJsonParser()
        return codeblock_parser.loads(response_text)

def generate_structure_char_mask(s: str) -> List[bool]:
    """
    JSON 텍스트 s의 각 문자에 대해:
      - value(문자열/숫자/bool/null) 구간은 False
      - 그 외(키, 구분자, 중괄호 등)는 True
    를 반환한다 (mask = [True]*n 으로 시작 → value에서 False로 전환).

    추가 규칙:
      - 콜론 뒤 값 처리
      - 배열이 '문자열/숫자/bool/null'로만 이루어진 리스트라면
        그 배열 전체(대괄호, 쉼표 포함)를 전부 False로 마스킹
      - 빈 배열 [] 및 빈 객체 {}는 각각 대괄호/중괄호를 False 처리
    """
    n = len(s)
    mask = [True] * n

    def set_false(a: int, b: int):
        a = max(0, a); b = min(n - 1, b)
        if a <= b:
            for i in range(a, b + 1):
                mask[i] = False

    def skip_ws(i: int) -> int:
        while i < n and s[i].isspace():
            i += 1
        return i

    def string_end(i: int) -> int:
        # i는 여는 따옴표('"') 위치
        j = i + 1
        while j < n:
            c = s[j]
            if c == "\\":
                j += 2          # 이스케이프 다음 글자 스킵
            elif c == '"':
                return j
            else:
                j += 1
        return n - 1             # 비정상 종료 시 끝까지

    number_regex = re.compile(r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?')

    def number_end(i: int) -> int:
        m = number_regex.match(s, i)
        return m.end() - 1 if m else i

    def literal_match(i: int, lit: str) -> bool:
        return s.startswith(lit, i)

    def primitive_only_array_end(i_lbrack: int):
        """
        s[i_lbrack] == '[' 인 위치에서,
        배열이 (문자열/숫자/true/false/null) 원소만으로 구성되어 있으면
        닫는 ']'의 인덱스를 반환. 아니면 None.
        공백과 쉼표는 허용.
        """
        i = skip_ws(i_lbrack + 1)
        if i >= n:
            return None
        # 빈 배열 []
        if s[i] == ']':
            return i

        while i < n:
            i = skip_ws(i)
            if i >= n:
                return None
            c = s[i]
            # 문자열 원소
            if c == '"':
                e = string_end(i)
                i = e + 1
            # 숫자 원소
            elif c in "-0123456789":
                e = number_end(i)
                i = e + 1
            # true/false/null
            elif literal_match(i, "true"):
                i += 4
            elif literal_match(i, "false"):
                i += 5
            elif literal_match(i, "null"):
                i += 4
            else:
                # 객체/배열/기타 토큰 등장 → primitive-only 아님
                return None

            i = skip_ws(i)
            if i >= n:
                return None
            if s[i] == ",":
                i += 1
                continue
            if s[i] == "]":
                return i  # 성공
            # 다른 토큰 → 실패
            return None

    prev_sig = None
    i = 0
    in_string = False

    while i < n:
        c = s[i]

        # 문자열 경계 관리(마스킹은 콜론/배열 진입 규칙에서 처리)
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
                prev_sig = '"'
            i += 1
            continue

        if c == '"':
            in_string = True
            i += 1
            continue

        if c.isspace():
            i += 1
            continue

        # ---------- 콜론 뒤 값 ----------
        if c == ":":
            j = skip_ws(i + 1)
            if j >= n:
                i += 1; prev_sig = ":"; continue
            nxt = s[j]

            # 배열 값
            if nxt == "[":
                arr_end = primitive_only_array_end(j)
                if arr_end is not None:
                    # 규칙 3: 원시값 전용 리스트 → 배열 전체 마스킹
                    set_false(j, arr_end)
                    i = arr_end + 1
                    prev_sig = "]"
                    continue
                # 빈 배열이 아니거나 원시 전용이 아니면, 그대로 진행(내부에서 다른 규칙으로 처리)
            # 객체 값
            if nxt == "{":
                j2 = skip_ws(j + 1)
                if j2 < n and s[j2] == "}":
                    # {} → 둘 다 False
                    set_false(j, j)     # '{'
                    set_false(j2, j2)   # '}'
                    i = j2 + 1
                    prev_sig = "}"
                    continue
                # 비어있지 않으면 일반 로직으로

            # 문자열 값
            if nxt == '"':
                e = string_end(j)
                set_false(j, e)
                i = e + 1
                prev_sig = '"'
                continue

            # 숫자 값
            if nxt in "-0123456789":
                e = number_end(j)
                set_false(j, e)
                i = e + 1
                prev_sig = s[e] if e < n else None
                continue

            # true / false / null
            if literal_match(j, "true"):
                set_false(j, j + 3)
                i = j + 4
                prev_sig = "e"
                continue
            if literal_match(j, "false"):
                set_false(j, j + 4)
                i = j + 5
                prev_sig = "e"
                continue
            if literal_match(j, "null"):
                set_false(j, j + 3)
                i = j + 4
                prev_sig = "l"
                continue

            # 그 외(객체/배열 등) → 내부에서 계속 처리
            i += 1
            prev_sig = ":"
            continue

        # ---------- 배열 원소 시작(prev_sig가 '[' 또는 ',') ----------
        # (배열이 값으로 오지 않아도, 최상위/중첩 배열 모두 커버)
        if prev_sig in ("[", ","):
            # 배열이 원시 전용이면 전체 마스킹(여기서도 인식)
            if c == "[":
                arr_end = primitive_only_array_end(i)
                if arr_end is not None:
                    set_false(i, arr_end)
                    i = arr_end + 1
                    prev_sig = "]"
                    continue
            # 개별 원소가 원시 리터럴이면 그 리터럴만 마스킹
            if c == '"':
                e = string_end(i)
                set_false(i, e)
                i = e + 1
                prev_sig = '"'
                continue
            if c in "-0123456789":
                e = number_end(i)
                set_false(i, e)
                i = e + 1
                prev_sig = s[e] if e < n else None
                continue
            if literal_match(i, "true"):
                set_false(i, i + 3)
                i = i + 4
                prev_sig = "e"
                continue
            if literal_match(i, "false"):
                set_false(i, i + 4)
                i = i + 5
                prev_sig = "e"
                continue
            if literal_match(i, "null"):
                set_false(i, i + 3)
                i = i + 4
                prev_sig = "l"
                continue

        # 구조 문자/구분자 등 prev_sig 갱신
        if c in "{}[],":
            prev_sig = c
        else:
            prev_sig = c

        i += 1

    return mask
