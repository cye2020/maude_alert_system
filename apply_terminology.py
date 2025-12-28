#!/usr/bin/env python3
"""
용어 통일 적용 스크립트

하드코딩된 한글을 찾아서 Terms로 변경하는 것을 돕는 스크립트
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# 하드코딩된 한글 패턴
HARDCODED_PATTERNS = {
    # 메트릭
    r'"치명률"': 'Terms.KOREAN.CFR',
    r"'치명률'": 'Terms.KOREAN.CFR',
    r'"치명률\(CFR\)"': 'Terms.KOREAN.CFR_FULL',
    r"'치명률\(CFR\)'": 'Terms.KOREAN.CFR_FULL',
    r'"사망률"': 'Terms.KOREAN.DEATH_RATE',
    r"'사망률'": 'Terms.KOREAN.DEATH_RATE',
    r'"사망"': 'Terms.KOREAN.DEATH_COUNT',
    r"'사망'": 'Terms.KOREAN.DEATH_COUNT',
    r'"중대 피해"': 'Terms.KOREAN.SEVERE_HARM',
    r"'중대 피해'": 'Terms.KOREAN.SEVERE_HARM',
    r'"중증 부상"': 'Terms.KOREAN.SERIOUS_INJURY',
    r"'중증 부상'": 'Terms.KOREAN.SERIOUS_INJURY',
    r'"보고 건수"': 'Terms.KOREAN.REPORT_COUNT',
    r"'보고 건수'": 'Terms.KOREAN.REPORT_COUNT',

    # 엔티티
    r'"제조사"': 'Terms.KOREAN.MANUFACTURER',
    r"'제조사'": 'Terms.KOREAN.MANUFACTURER',
    r'"제품군"': 'Terms.KOREAN.PRODUCT',
    r"'제품군'": 'Terms.KOREAN.PRODUCT',
    r'"기기"': 'Terms.KOREAN.DEVICE',
    r"'기기'": 'Terms.KOREAN.DEVICE',
    r'"결함 유형"': 'Terms.KOREAN.DEFECT_TYPE',
    r"'결함 유형'": 'Terms.KOREAN.DEFECT_TYPE',
    r'"문제 부품"': 'Terms.KOREAN.COMPONENT',
    r"'문제 부품'": 'Terms.KOREAN.COMPONENT',
    r'"클러스터"': 'Terms.KOREAN.CLUSTER',
    r"'클러스터'": 'Terms.KOREAN.CLUSTER',

    # 패턴
    r'"급증"': 'Terms.KOREAN.SPIKE',
    r"'급증'": 'Terms.KOREAN.SPIKE',
    r'"증가"': 'Terms.KOREAN.INCREASE',
    r"'증가'": 'Terms.KOREAN.INCREASE',
    r'"감소"': 'Terms.KOREAN.DECREASE',
    r"'감소'": 'Terms.KOREAN.DECREASE',

    # 시간
    r'"시계열"': 'Terms.KOREAN.TIME_SERIES',
    r"'시계열'": 'Terms.KOREAN.TIME_SERIES',
    r'"추이"': 'Terms.KOREAN.TREND',
    r"'추이'": 'Terms.KOREAN.TREND',
    r'"월별"': 'Terms.KOREAN.MONTHLY',
    r"'월별'": 'Terms.KOREAN.MONTHLY',

    # 분석
    r'"분포"': 'Terms.KOREAN.DISTRIBUTION',
    r"'분포'": 'Terms.KOREAN.DISTRIBUTION',

    # 섹션
    r'"개요"': 'Terms.KOREAN.OVERVIEW',
    r"'개요'": 'Terms.KOREAN.OVERVIEW',
    r'"요약"': 'Terms.KOREAN.SUMMARY',
    r"'요약'": 'Terms.KOREAN.SUMMARY',
    r'"인사이트"': 'Terms.KOREAN.INSIGHTS',
    r"'인사이트'": 'Terms.KOREAN.INSIGHTS',

    # 분석 섹션
    r'"결함 유형 분석"': 'Terms.KOREAN.DEFECT_TYPE_ANALYSIS',
    r"'결함 유형 분석'": 'Terms.KOREAN.DEFECT_TYPE_ANALYSIS',
    r'"문제 부품 분석"': 'Terms.KOREAN.COMPONENT_ANALYSIS',
    r"'문제 부품 분석'": 'Terms.KOREAN.COMPONENT_ANALYSIS',
    r'"환자 피해 분포"': 'Terms.KOREAN.HARM_DISTRIBUTION',
    r"'환자 피해 분포'": 'Terms.KOREAN.HARM_DISTRIBUTION',
    r'"사건 유형 분포"': 'Terms.KOREAN.EVENT_TYPE_DISTRIBUTION',
    r"'사건 유형 분포'": 'Terms.KOREAN.EVENT_TYPE_DISTRIBUTION',
    r'"치명률\(CFR\) 분석"': 'Terms.KOREAN.CFR_ANALYSIS',
    r"'치명률\(CFR\) 분석'": 'Terms.KOREAN.CFR_ANALYSIS',
    r'"리스크 매트릭스"': 'Terms.KOREAN.RISK_MATRIX',
    r"'리스크 매트릭스'": 'Terms.KOREAN.RISK_MATRIX',
}


def find_hardcoded_strings(file_path: Path) -> List[Tuple[int, str, str]]:
    """파일에서 하드코딩된 한글 문자열 찾기

    Returns:
        List of (line_number, line_content, matched_pattern)
    """
    matches = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            # streamlit 관련 함수에서만 찾기
            if any(keyword in line for keyword in ['st.metric', 'st.subheader', 'st.markdown', 'st.title', 'name=']):
                # 하드코딩된 패턴 찾기
                for pattern in HARDCODED_PATTERNS.keys():
                    if re.search(pattern, line):
                        matches.append((i, line.strip(), pattern))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return matches


def scan_dashboard_files(dashboard_dir: Path = None):
    """대시보드 파일들 스캔"""
    if dashboard_dir is None:
        dashboard_dir = Path(__file__).parent / 'dashboard'

    print("=" * 80)
    print("하드코딩된 한글 문자열 검색")
    print("=" * 80)

    py_files = list(dashboard_dir.glob('**/*.py'))

    total_matches = 0
    files_with_issues = []

    for py_file in sorted(py_files):
        # __pycache__ 제외
        if '__pycache__' in str(py_file):
            continue

        matches = find_hardcoded_strings(py_file)

        if matches:
            files_with_issues.append((py_file, matches))
            total_matches += len(matches)

    # 결과 출력
    if files_with_issues:
        print(f"\n총 {len(files_with_issues)}개 파일에서 {total_matches}개 하드코딩 발견\n")

        for file_path, matches in files_with_issues:
            rel_path = file_path.relative_to(Path.cwd())
            print(f"\n📄 {rel_path}")
            print("-" * 80)

            for line_num, line_content, pattern in matches:
                replacement = HARDCODED_PATTERNS[pattern]
                print(f"  Line {line_num:4d}: {line_content[:70]}")
                print(f"             → {pattern} => {replacement}")
                print()
    else:
        print("\n✅ 하드코딩된 문자열이 없습니다!")

    return files_with_issues


def show_migration_tips():
    """마이그레이션 팁 표시"""
    print("\n" + "=" * 80)
    print("📚 마이그레이션 가이드")
    print("=" * 80)
    print("""
1. 임포트 추가:
   from dashboard.utils.constants import Terms

2. 간단한 변경:
   "치명률" → Terms.KOREAN.CFR
   "사망" → Terms.KOREAN.DEATH_COUNT

3. f-string에서 사용:
   st.subheader(f"📈 {Terms.KOREAN.REPORT_COUNT} {Terms.KOREAN.TREND}")

4. 템플릿 사용:
   st.subheader(Terms.section_title('entity_analysis', entity=Terms.KOREAN.DEFECT_TYPE))

5. 상세 가이드:
   MIGRATION_GUIDE.md 참고
""")


if __name__ == '__main__':
    print("\n🔍 대시보드 용어 통일 검사 도구\n")

    # 스캔 실행
    issues = scan_dashboard_files()

    # 팁 표시
    if issues:
        show_migration_tips()

        print("\n" + "=" * 80)
        print("다음 단계:")
        print("=" * 80)
        print("1. MIGRATION_GUIDE.md 읽기")
        print("2. 파일별로 하드코딩 → Terms로 변경")
        print("3. 테스트 후 커밋")
        print()
