import re
import json
import jsonschema
from typing import Literal

import torch


class ParserError(Exception):
    pass


class ValidationError(Exception):
    pass


JSON_REGEX = re.compile(r"```json\s*([\s\S]*?)\s*```", re.IGNORECASE)

# for GPT2 tokenizer
JSON_STRUCTURE_TOKEN_IDS = [1, 11, 25, 58, 60, 90, 92, 553, 1298, 1600, 2404, 2430, 3712, 4357, 4895, 5512, 5974, 7131, 8351, 8762, 8973, 9063, 9832, 11097, 11709, 11907, 11919, 13018, 14692, 15931, 17241, 17414, 17912, 18477, 20598, 20662, 21737, 23846, 24022, 25719, 26358, 27007, 29164, 30109, 30866, 32509, 33116, 33250, 34171, 34713, 36786, 37811, 38362, 38430, 42535, 42785, 43661, 45299, 47182, 47682, 47715, 48999]


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
