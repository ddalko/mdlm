#!/usr/bin/env python3
"""
JSON Error Analyzer
콤마(,)와 콜론(:) 구분자 오류를 포함한 JSON 분석 및 시각화 도구
"""

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ErrorInfo:
    position: Optional[int]
    error_type: str
    description: str
    snippet: str

class JSONErrorAnalyzer:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = []
        self.load_data()
    
    def load_data(self):
        """JSON 파일 로드"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"데이터 로드 완료: {len(self.data)}개 항목")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return
    
    def extract_error_info(self, error_msg: str) -> ErrorInfo:
        """에러 메시지에서 정보 추출"""
        if not error_msg:
            return ErrorInfo(None, "no_error", "오류 없음", "")
        
        # 위치 정보 추출
        pos_match = re.search(r'at pos (\d+):', error_msg)
        position = int(pos_match.group(1)) if pos_match else None
        
        # 스니펫 추출
        snippet_match = re.search(r'Snippet: (.+)$', error_msg)
        snippet = snippet_match.group(1) if snippet_match else ""
        
        # 에러 타입 분류
        if "Expecting ',' delimiter" in error_msg:
            error_type = "missing_comma"
            description = "콤마(,) 누락"
        elif "Expecting ':' delimiter" in error_msg:
            error_type = "missing_colon"
            description = "콜론(:) 누락"
        elif "Expecting value" in error_msg:
            error_type = "expecting_value"
            description = "값 없음"
        elif "Invalid control character" in error_msg:
            error_type = "invalid_control_char"
            description = "잘못된 제어 문자"
        elif "Invalid \\escape" in error_msg:
            error_type = "invalid_escape"
            description = "잘못된 이스케이프"
        elif "Unterminated string" in error_msg:
            error_type = "unterminated_string"
            description = "종료되지 않은 문자열"
        elif "Extra data" in error_msg:
            error_type = "extra_data"
            description = "추가 데이터"
        else:
            error_type = "other"
            description = "기타 오류"
            
        return ErrorInfo(position, error_type, description, snippet)
    
    def analyze_type_mismatch(self, pred_json: str) -> List[str]:
        """타입 불일치 분석 (숫자/boolean 자리에 문자열이 온 경우)"""
        issues = []
        
        # 숫자가 와야 할 자리에 문자열이 온 경우 패턴
        number_patterns = [
            r':\s*"[^"]*"(?=\s*[,}])',  # 따옴표로 둘러싸인 값
            r':\s*[a-zA-Z][a-zA-Z0-9_]*(?=\s*[,}])',  # 따옴표 없는 문자열
        ]
        
        for pattern in number_patterns:
            matches = list(re.finditer(pattern, pred_json))
            for match in matches:
                value = match.group().split(':')[1].strip()
                
                # 숫자처럼 보이지만 문자열인 경우
                if (value.startswith('"') and value.endswith('"')):
                    inner_value = value[1:-1]
                    if re.match(r'^\d+\.?\d*$', inner_value) or inner_value.lower() in ['true', 'false', 'null']:
                        issues.append(f"타입 불일치: '{value}' (위치: {match.start()})")
                # boolean이나 null이 따옴표 없이 문자열로 온 경우
                elif value.lower() in ['true', 'false', 'null'] and not value.startswith('"'):
                    issues.append(f"타입 불일치: '{value}' (위치: {match.start()})")
        
        return issues
    
    def highlight_json_structure(self, text: str) -> str:
        """JSON 구조 문자를 노란색으로 하이라이트"""
        # JSON 구조 문자들
        structure_chars = {'{', '}', '[', ']', '"', ',', ':'}
        
        result = []
        for char in text:
            if char in structure_chars:
                escaped_char = html.escape(char)
                result.append(f'<span style="background-color: #fff3cd; color: #856404; font-weight: bold;">{escaped_char}</span>')
            else:
                result.append(html.escape(char))
        
        return ''.join(result)
    
    def highlight_error_position(self, text: str, position: int, context_chars: int = 50) -> str:
        """에러 위치를 빨간색으로 하이라이트하고 구조 문자는 노란색으로 하이라이트"""
        if position is None or position >= len(text):
            return self.highlight_json_structure(text)
        
        # 먼저 구조 문자를 하이라이트
        structure_chars = {'{', '}', '[', ']', '"', ',', ':'}
        
        result = []
        for i, char in enumerate(text):
            if i == position:
                # 에러 위치는 빨간색으로 하이라이트 (구조 문자보다 우선)
                escaped_char = html.escape(char)
                result.append(f'<span style="background-color: #ff6b6b; color: white; font-weight: bold; padding: 2px;">{escaped_char}</span>')
            elif char in structure_chars:
                # 구조 문자는 노란색으로 하이라이트
                escaped_char = html.escape(char)
                result.append(f'<span style="background-color: #fff3cd; color: #856404; font-weight: bold;">{escaped_char}</span>')
            else:
                result.append(html.escape(char))
        
        return ''.join(result)
    
    def generate_html_report(self) -> str:
        """HTML 리포트 생성"""
        error_items = []
        error_stats = {}
        type_mismatch_stats = {}
        
        for item in self.data:
            error_msg = item.get('error_msg', '')
            if not error_msg:
                continue
                
            error_info = self.extract_error_info(error_msg)
            
            # 통계 업데이트
            error_stats[error_info.error_type] = error_stats.get(error_info.error_type, 0) + 1
            
            # 타입 불일치 분석
            type_mismatches = self.analyze_type_mismatch(item['pred'])
            for mismatch in type_mismatches:
                mismatch_type = mismatch.split(':')[1].strip().split()[0]
                type_mismatch_stats[mismatch_type] = type_mismatch_stats.get(mismatch_type, 0) + 1
            
            # GT와 pred 비교를 위한 하이라이팅
            gt_highlighted = self.highlight_json_structure(item['gt'])
            pred_highlighted = self.highlight_error_position(item['pred'], error_info.position)
            
            error_items.append({
                'id': item['id'],
                'error_info': error_info,
                'gt': gt_highlighted,
                'pred': pred_highlighted,
                'type_mismatches': type_mismatches
            })
        
        # HTML 생성
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JSON 오류 분석 리포트</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .stat-card h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .error-item {{
            background: white;
            margin-bottom: 20px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid #ff6b6b;
        }}
        
        .error-header {{
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .error-id {{
            font-weight: bold;
            color: #495057;
        }}
        
        .error-type {{
            background: #ff6b6b;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        
        .json-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }}
        
        .json-section {{
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
        }}
        
        .json-section h4 {{
            margin-top: 0;
            color: #495057;
        }}
        
        .json-content {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        
        .type-mismatches {{
            padding: 0 20px 20px 20px;
        }}
        
        .mismatch-item {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 8px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        .filters {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .filter-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .filter-button {{
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .filter-button.active {{
            background: #667eea;
            color: white;
        }}
        
        @media (max-width: 768px) {{
            .json-comparison {{
                grid-template-columns: 1fr;
            }}
            
            .stats {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>JSON 오류 분석 리포트</h1>
        <p>콤마(,)와 콜론(:) 구분자 오류 및 타입 불일치 분석</p>
        <p>총 {len(error_items)}개의 오류 발견 (전체 {len(self.data)}개 중)</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>📊 오류 유형별 통계</h3>
            {self._generate_stats_html(error_stats)}
        </div>
        
        <div class="stat-card">
            <h3>🔧 타입 불일치 통계</h3>
            {self._generate_stats_html(type_mismatch_stats)}
        </div>
    </div>
    
    <div class="filters">
        <h3>필터</h3>
        <div class="filter-group">
            <button class="filter-button active" onclick="filterErrors('all')">전체</button>
            <button class="filter-button" onclick="filterErrors('missing_comma')">콤마 누락</button>
            <button class="filter-button" onclick="filterErrors('missing_colon')">콜론 누락</button>
            <button class="filter-button" onclick="filterErrors('expecting_value')">값 없음</button>
            <button class="filter-button" onclick="filterErrors('invalid_control_char')">제어 문자</button>
            <button class="filter-button" onclick="filterErrors('invalid_escape')">이스케이프</button>
            <button class="filter-button" onclick="filterErrors('other')">기타</button>
        </div>
    </div>
    
    <div id="error-list">
        {self._generate_error_items_html(error_items)}
    </div>
    
    <script>
        function filterErrors(type) {{
            // 버튼 상태 업데이트
            document.querySelectorAll('.filter-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // 에러 아이템 필터링
            document.querySelectorAll('.error-item').forEach(item => {{
                const errorType = item.dataset.errorType;
                if (type === 'all' || errorType === type) {{
                    item.style.display = 'block';
                }} else {{
                    item.style.display = 'none';
                }}
            }});
        }}
        
        // 페이지 로드 시 통계 업데이트
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('JSON 오류 분석 리포트 로드 완료');
        }});
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _generate_stats_html(self, stats: Dict[str, int]) -> str:
        """통계 HTML 생성"""
        if not stats:
            return "<p>데이터 없음</p>"
        
        total = sum(stats.values())
        html_items = []
        
        for error_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            html_items.append(f"""
                <div class="stat-item">
                    <span>{error_type}</span>
                    <span>{count}개 ({percentage:.1f}%)</span>
                </div>
            """)
        
        return "".join(html_items)
    
    def _generate_error_items_html(self, error_items: List[Dict]) -> str:
        """에러 아이템 HTML 생성"""
        html_items = []
        
        for item in error_items:
            type_mismatches_html = ""
            if item['type_mismatches']:
                mismatch_items = "".join([f'<div class="mismatch-item">{html.escape(tm)}</div>' 
                                        for tm in item['type_mismatches']])
                type_mismatches_html = f"""
                    <div class="type-mismatches">
                        <h4>🚨 타입 불일치</h4>
                        {mismatch_items}
                    </div>
                """
            
            html_items.append(f"""
                <div class="error-item" data-error-type="{item['error_info'].error_type}">
                    <div class="error-header">
                        <span class="error-id">ID: {item['id']}</span>
                        <span class="error-type">{item['error_info'].description}</span>
                        {f'<span style="margin-left: 10px; color: #666;">위치: {item["error_info"].position}</span>' if item['error_info'].position else ''}
                    </div>
                    
                    <div class="json-comparison">
                        <div class="json-section">
                            <h4>🎯 정답 (GT)</h4>
                            <div class="json-content">{item['gt']}</div>
                        </div>
                        
                        <div class="json-section">
                            <h4>🤖 예측 (Pred)</h4>
                            <div class="json-content">{item['pred']}</div>
                        </div>
                    </div>
                    
                    {type_mismatches_html}
                </div>
            """)
        
        return "".join(html_items)
    
    def generate_report(self, output_file: str = "json_error_analysis.html"):
        """리포트 생성 및 저장"""
        html_content = self.generate_html_report()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"리포트가 생성되었습니다: {output_file}")
            return output_file
        except Exception as e:
            print(f"리포트 생성 실패: {e}")
            return None

def main():
    """메인 함수"""
    json_file = "/workspace/mdlm/outputs/schemabench/2025.09.24/122816/json_eval_results.json"
    output_file = "/workspace/mdlm/json_error_analysis.html"
    
    print("JSON 오류 분석 시작...")
    analyzer = JSONErrorAnalyzer(json_file)
    
    if analyzer.data:
        report_file = analyzer.generate_report(output_file)
        if report_file:
            print(f"\n✅ 분석 완료!")
            print(f"📄 리포트 파일: {report_file}")
            print(f"🌐 브라우저에서 열어보세요!")
        else:
            print("❌ 리포트 생성에 실패했습니다.")
    else:
        print("❌ 데이터 로드에 실패했습니다.")

if __name__ == "__main__":
    main()