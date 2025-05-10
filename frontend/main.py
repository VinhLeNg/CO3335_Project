import streamlit as st
import requests
import os
import json
import io

from audiorecorder import audiorecorder

# CONFIG
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
STATUS_ENDPOINT = f"{BACKEND_URL}/status"
STT_ENDPOINT = f"{BACKEND_URL}/speech-to-text"

def get_chatbot_response(query: str):
    """Sends query to backend and returns the response."""
    payload = {"query": query}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=360)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Kết nối tới backend chat thất bại: {e}", icon="🚨")
        st.error(f"Hãy đảm bảo backend đang chạy tại {BACKEND_URL} và Ollama đang hoạt động.", icon="⚙️")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã phản hồi JSON từ backend.", icon="📄")
        st.error(f"Phản hồi nhận được (đoạn đầu): {response.text[:200]}...", icon="📄")
        return None

def get_transcription(audio_bytes: bytes, filename: str = "audio.wav"):
    """Sends audio to backend for transcription."""
    if not audio_bytes:
        st.warning("Không có dữ liệu âm thanh để gửi.")
        return None
    files = {'audio_file': (filename, audio_bytes, 'audio/wav')}
    try:
        st.info("Đang gửi âm thanh đến backend để nhận diện...")
        response = requests.post(STT_ENDPOINT, files=files, timeout=120)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            st.error(f"Lỗi nhận diện giọng nói từ backend: {data['error']}", icon="🔊")
            return None
        transcribed_text = data.get("transcribed_text")
        if transcribed_text is not None:
             st.success("Nhận diện giọng nói thành công!")
             return transcribed_text
        else:
            st.error("Không nhận được văn bản từ backend STT.", icon="📄")
            return None
    except requests.exceptions.Timeout:
        st.error(f"Yêu cầu nhận diện giọng nói bị hết thời gian chờ...", icon="⏳")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Kết nối tới backend STT thất bại: {e}", icon="🚨")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã phản hồi JSON từ backend STT.", icon="📄")
        st.error(f"Phản hồi STT nhận được (đoạn đầu): {response.text[:200]}...", icon="📄")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không mong muốn khi nhận diện giọng nói: {e}", icon="💥")
        return None

# Page Config
st.set_page_config(layout="wide", page_title="UTC Chatbot")

# Title and Caption
st.title("ChatBot -- Trường Đại học giao thông vận tải")
st.caption("Nhập câu hỏi hoặc sử dụng giọng nói để nhận câu trả lời từ Trường ĐH GTVT.")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Bạn cần hỏi thông tin gì về Trường ĐH GTVT?"}
    ]
if "stt_prompt_to_send" not in st.session_state:
    st.session_state.stt_prompt_to_send = None
if "audio_submitted_for_stt" not in st.session_state:
    st.session_state.audio_submitted_for_stt = False

st.markdown("---")
st.subheader("Hoặc hỏi bằng giọng nói:")

audio = audiorecorder("Nhấn để ghi âm", "Dừng ghi âm", key="audiorec_component")
transcribed_text_from_audio = None

if audio and len(audio) > 0 and not st.session_state.audio_submitted_for_stt:
    st.info("Đã ghi âm xong. Đang xử lý...")
    
    audio_display_bytes_io = io.BytesIO()
    try:
        audio.export(audio_display_bytes_io, format="wav")
        st.audio(audio_display_bytes_io.getvalue(), format="audio/wav")

        with st.spinner("Đang nhận diện giọng nói... Xin chờ một chút..."):
            audio_stt_bytes_io = io.BytesIO()
            audio.export(audio_stt_bytes_io, format="wav")
            transcribed_text_from_audio = get_transcription(audio_stt_bytes_io.getvalue())

        if transcribed_text_from_audio:
            st.info(f"**Văn bản nhận diện được:** \"{transcribed_text_from_audio}\"")
            if st.button("📝 Gửi câu hỏi này", key="send_stt_text_button"):
                st.session_state.stt_prompt_to_send = transcribed_text_from_audio
                st.session_state.audio_submitted_for_stt = True # Mark as submitted
                st.rerun() # Use st.rerun()
        else:
            st.warning("Không thể nhận diện giọng nói từ đoạn ghi âm vừa rồi.")
            
    except Exception as e:
        st.error(f"Lỗi xử lý audio từ audiorecorder: {e}")
        st.error("Hãy đảm bảo 'ffmpeg' đã được cài đặt trong môi trường frontend nếu lỗi liên quan đến export/conversion.")

if st.session_state.audio_submitted_for_stt and st.session_state.stt_prompt_to_send is None:
    st.session_state.audio_submitted_for_stt = False

st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
             with st.expander("Xem nội dung tài liệu được sử dụng"):
                for i, chunk_info in enumerate(message["sources"]):
                     st.markdown(f"**Đoạn trích {i+1} (Từ: {chunk_info.get('source', 'Không rõ')})**")
                     st.markdown(f"> {chunk_info.get('content', 'N/A')}")

user_typed_prompt = st.chat_input("Đặt câu hỏi của bạn tại đây...")
final_prompt = None

if user_typed_prompt:
    final_prompt = user_typed_prompt
    if st.session_state.stt_prompt_to_send:
        st.session_state.stt_prompt_to_send = None
    st.session_state.audio_submitted_for_stt = False

elif st.session_state.stt_prompt_to_send:
    final_prompt = st.session_state.stt_prompt_to_send

if final_prompt:
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        message_placeholder.markdown("Đang suy nghĩ...") 
        response_data = get_chatbot_response(final_prompt)

        if response_data:
            answer = response_data.get("answer", "Xin lỗi, tôi không thể tạo câu trả lời.")
            source_chunks = response_data.get("source_chunks", [])
            message_placeholder.markdown(answer)
            if source_chunks:
                with st.expander("Xem nội dung tài liệu được sử dụng"):
                    for i, chunk_info in enumerate(source_chunks):
                        st.markdown(f"**Đoạn trích {i+1} (Từ: {chunk_info.get('source', 'Không rõ')})**")
                        st.markdown(f"> {chunk_info.get('content', 'N/A')}") 
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_chunks 
            })
        else:
            error_msg = "Xin lỗi, đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            message_placeholder.markdown(error_msg)
            is_api_error_already_logged = False
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                pass
            
            if not is_api_error_already_logged and (not st.session_state.messages or st.session_state.messages[-1]["content"] != error_msg):
                 st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    if st.session_state.stt_prompt_to_send == final_prompt:
        st.session_state.stt_prompt_to_send = None
else:
    if not st.session_state.stt_prompt_to_send:
        st.session_state.audio_submitted_for_stt = False
