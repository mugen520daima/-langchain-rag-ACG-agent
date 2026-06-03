"""Streamlit 页面入口"""
import streamlit as st
from agent_service import AnimeAgent

#启动命令：source venv/bin/activate && streamlit run app.py
# http://localhost:8557/
st.set_page_config(page_title="Weller的ACG助手", page_icon="🎌")
st.title("🎌 Weller的ACG助手")
st.caption("查询番剧资料、周边信息、正版观看渠道")

if "agent" not in st.session_state:
    st.session_state.agent = AnimeAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("问我关于动漫的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.agent.chat(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
