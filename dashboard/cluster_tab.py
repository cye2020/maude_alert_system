# cluster_tab.py
import streamlit as st

def show():
    st.session_state.current_tab = "Cluster"
    st.header("Cluster Reports")

    # ==================== 최근 활동 ====================
    st.subheader("📝 최근 활동")

    with st.expander("최근 24시간 활동 내역", expanded=True):
        # 샘플 활동 데이터
        activities = [
            {"time": "2시간 전", "event": "데이터 전처리 완료", "status": "✅"},
            {"time": "5시간 전", "event": "모델 학습 시작", "status": "🔄"},
            {"time": "8시간 전", "event": "새 데이터 수집 (1,500건)", "status": "✅"},
            {"time": "12시간 전", "event": "배치 작업 완료", "status": "✅"},
        ]
        
        for activity in activities:
            col1, col2, col3 = st.columns([1, 5, 1])
            with col1:
                st.markdown(f"**{activity['time']}**")
            with col2:
                st.markdown(activity['event'])
            with col3:
                st.markdown(activity['status'])

    st.markdown("---")