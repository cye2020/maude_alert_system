# Sidebar Enabled 데코레이터 개선

## 목적

`sidebar.yaml`의 `enabled` 플래그를 일관되게 처리하기 위해 `@check_enabled` 데코레이터를 도입했습니다.

## 문제점

기존에는 각 메서드마다 수동으로 `enabled` 체크를 해야 했습니다:

```python
def render_date_selector(self):
    date_config = self.common_config.get("date_selector", {})

    # ❌ 수동 체크 - 누락될 수 있음
    if not date_config.get("enabled", False):
        return None

    # 실제 로직...
```

**문제점**:
- 모든 메서드에 반복적인 코드
- 실수로 체크를 누락할 가능성
- 일관되지 않은 기본값 처리 (`True` vs `False`)

## 해결책: `@check_enabled` 데코레이터

### 구현

```python
def check_enabled(config_path: str):
    """enabled 플래그를 체크하는 데코레이터

    Args:
        config_path: 체크할 설정 경로 (예: 'common.header', 'common.date_selector')

    Returns:
        enabled=False면 None을 반환, True면 원래 함수 실행
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # config_path를 따라 설정 탐색
            parts = config_path.split('.')
            config = self.cfg.sidebar

            for part in parts:
                config = config.get(part, {})
                if not config:
                    # 설정이 없으면 기본적으로 활성화
                    return func(self, *args, **kwargs)

            # enabled 체크
            if not config.get('enabled', True):
                return None

            return func(self, *args, **kwargs)

        return wrapper
    return decorator
```

### 사용 예시

#### Before (수동 체크)

```python
def render_header(self):
    """프로젝트 로고 및 정보 렌더링"""
    header_config = self.common_config.get("header", {})

    # ❌ 수동으로 enabled 체크해야 함
    if not header_config.get("enabled", True):
        return None

    # 로고 렌더링...
    logo_path = header_config.get("logo")
    if logo_path:
        st.image(logo_path, width='stretch')
```

#### After (데코레이터 사용)

```python
@check_enabled('common.header')
def render_header(self):
    """프로젝트 로고 및 정보 렌더링"""
    header_config = self.common_config.get("header", {})

    # ✅ 데코레이터가 자동으로 enabled 체크
    # enabled=False면 함수가 실행되지 않음

    # 로고 렌더링...
    logo_path = header_config.get("logo")
    if logo_path:
        st.image(logo_path, width='stretch')
```

## sidebar.yaml 설정

```yaml
common:
  header:
    enabled: true  # false로 설정 시 render_header() 실행 안 됨
    logo: "dashboard/assets/logo.png"

  date_selector:
    enabled: false  # render_date_selector() 실행 안 됨
```

## 적용된 메서드

### SidebarManager 클래스

1. **`render_header()`**
   - 데코레이터: `@check_enabled('common.header')`
   - 설정 경로: `sidebar.yaml > common > header > enabled`

2. **`render_date_selector()`**
   - 데코레이터: `@check_enabled('common.date_selector')`
   - 설정 경로: `sidebar.yaml > common > date_selector > enabled`

3. **`render_widget()`**
   - 기존 방식 유지 (각 위젯의 `enabled` 플래그를 개별 체크)
   - 이유: 위젯은 동적으로 생성되므로 config_path를 미리 알 수 없음

## 장점

### 1. **코드 중복 제거**
   - 모든 메서드에 반복되는 `if not config.get("enabled")` 코드 제거

### 2. **일관성 보장**
   - 기본값이 항상 `True` (설정 없으면 활성화)
   - 모든 메서드가 동일한 로직 사용

### 3. **가독성 향상**
   - 메서드 정의부에서 한눈에 enabled 체크 여부 확인 가능
   - 비즈니스 로직과 설정 체크 분리

### 4. **유지보수성**
   - enabled 체크 로직 변경 시 데코레이터만 수정하면 됨
   - 새로운 메서드 추가 시 데코레이터만 붙이면 됨

## 설계 철학

### Dot Notation Path

데코레이터는 **dot notation 경로**를 사용합니다:

```python
@check_enabled('common.header')           # common > header > enabled
@check_enabled('common.date_selector')     # common > date_selector > enabled
@check_enabled('dashboards.overview.xyz')  # 향후 대시보드별 설정 지원
```

이는 YAML 구조를 그대로 반영하여 직관적입니다.

### 기본값: True

설정이 없거나 `enabled` 키가 없으면 **기본적으로 활성화**됩니다:

```python
# enabled 체크
if not config.get('enabled', True):  # ← 기본값 True
    return None
```

**이유**:
- Opt-out 방식: 명시적으로 비활성화하지 않으면 작동
- 설정 누락 시에도 안전하게 동작
- 기존 코드 호환성 유지

## 테스트

### 시나리오 1: enabled=true

```yaml
common:
  header:
    enabled: true
```

**결과**: `render_header()` 정상 실행 ✅

### 시나리오 2: enabled=false

```yaml
common:
  header:
    enabled: false
```

**결과**: `render_header()`가 `None` 반환하고 즉시 종료 ✅

### 시나리오 3: enabled 키 없음

```yaml
common:
  header:
    logo: "dashboard/assets/logo.png"
```

**결과**: 기본값 `True`로 처리되어 정상 실행 ✅

### 시나리오 4: 설정 자체가 없음

```yaml
common:
  # header 설정 없음
```

**결과**: 기본값 `True`로 처리되어 정상 실행 ✅

## 향후 확장

### 대시보드별 enabled 지원

현재는 `common.*` 경로만 지원하지만, 향후 대시보드별 설정도 가능:

```python
@check_enabled('dashboards.overview.specific_chart')
def render_specific_chart(self):
    # overview 탭에서만 활성화 여부 제어
    ...
```

```yaml
dashboards:
  overview:
    specific_chart:
      enabled: false  # overview 탭에서만 비활성화
```

### 조건부 enabled

더 복잡한 조건 지원 가능:

```python
@check_enabled('common.header', condition=lambda cfg: cfg.get('user_role') == 'admin')
def render_admin_panel(self):
    # admin 역할일 때만 활성화
    ...
```

## 관련 파일

- **구현**: [sidebar_manager.py:19-54](../utils/sidebar_manager.py#L19-L54)
- **사용**: [sidebar_manager.py:75-103](../utils/sidebar_manager.py#L75-L103)
- **설정**: [sidebar.yaml:18-28](../../config/dashboard/sidebar.yaml#L18-L28)
- **문서**: 본 문서

---

**작성일**: 2025-12-28
**버전**: v1.0.0
**담당**: Claude Code Refactoring Agent
