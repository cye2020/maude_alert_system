# eda_tab.py
import streamlit as st

def show(filters=None):
    st.title("📈 Detailed Analysis")

    # 필터 값 사용
    selected_date = filters.get("date")
    categories = filters.get("categories", [])
    confidence_interval = filters.get("confidence_interval", 0.95)
    
    # ==================== 주요 기능 안내 ====================
    st.subheader("📚 주요 기능")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📊 데이터 개요")
            st.markdown("""
            - Bronze/Silver/Gold 데이터 레이어 현황
            - 데이터 품질 지표
            - 최근 업데이트 이력
            """)
            if st.button("데이터 개요 보기", key="btn_data", width='stretch'):
                st.switch_page("pages/1_📊_Data_Overview.py")

    with col2:
        with st.container(border=True):
            st.markdown("### 📈 분석 대시보드")
            st.markdown("""
            - 인터랙티브 차트 및 시각화
            - 트렌드 분석
            - 커스텀 필터링
            """)
            if st.button("분석 대시보드 보기", key="btn_analytics", width='stretch'):
                st.switch_page("pages/2_📈_Analytics.py")

    col3, col4 = st.columns(2)

    with col3:
        with st.container(border=True):
            st.markdown("### 🤖 모델 성능")
            st.markdown("""
            - 모델 정확도 및 성능 지표
            - 학습 이력
            - A/B 테스트 결과
            """)
            if st.button("모델 성능 보기", key="btn_model", width='stretch'):
                st.switch_page("pages/3_🤖_Model_Performance.py")

    with col4:
        with st.container(border=True):
            st.markdown("### ⚙️ 설정")
            st.markdown("""
            - 데이터 소스 설정
            - 알림 설정
            - 사용자 권한 관리
            """)
            if st.button("설정 보기", key="btn_settings", width='stretch'):
                st.switch_page("pages/4_⚙️_Settings.py")

    st.markdown("---")