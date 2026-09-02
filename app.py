import os
import tempfile
import zipfile
import pymupdf4llm
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(layout="wide", page_title="AI Scientific Translator")
st.title("Chuyển ngữ Tài liệu Khoa học (RU/EN $\\rightarrow$ VN)")

with st.sidebar:
    st.header("Cấu hình hệ thống")
    api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
    source_lang = st.selectbox("Ngôn ngữ tài liệu gốc:", ["Tiếng Nga", "Tiếng Anh"])
    model_choice = st.selectbox("Tốc độ / Chất lượng:", ["gemini-3.6-flash", "gemini-3.1-pro"])
    
    st.markdown("---")
    st.markdown("**Từ điển chuyên ngành**")
    glossary = st.text_area(
        "Quy tắc dịch thuật nội bộ:", 
        value="displacement = chuyển vị\ntopology optimization = tối ưu hóa topo\ncompliance = độ tuân thủ"
    )

uploaded_file = st.file_uploader("Tải lên tài liệu (PDF)", type=["pdf"])

if uploaded_file and api_key:
    if st.button("Bắt đầu Dịch thuật", type="primary"):
        client = genai.Client(api_key=api_key)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Đang bóc tách PDF và trích xuất hình ảnh..."):
                image_dir = "images"
                os.makedirs(image_dir, exist_ok=True)
                md_text = pymupdf4llm.to_markdown(tmp_path, write_images=True, image_path=image_dir)
                
            with st.spinner("Đang xử lý dịch thuật (áp dụng Smart Chunking)..."):
                # Chia khối 10,000 ký tự để tránh nghẽn Output Limit
                paragraphs = md_text.split('\n\n')
                chunks = []
                current_chunk = ""
                for p in paragraphs:
                    if len(current_chunk) + len(p) < 10000:
                        current_chunk += p + "\n\n"
                    else:
                        chunks.append(current_chunk)
                        current_chunk = p + "\n\n"
                if current_chunk:
                    chunks.append(current_chunk)

                translated_text = ""
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Đang dịch phần {i+1}/{len(chunks)}...")
                    prompt = f"""
                    Bạn là chuyên gia dịch thuật tài liệu khoa học hàng không.
                    Dịch văn bản sau từ {source_lang} sang Tiếng Việt.
                    
                    YÊU CẦU BẮT BUỘC:
                    1. Giữ nguyên 100% thẻ LaTeX toán học ($...$, $$...$$), KHÔNG dịch biến số.
                    2. Tuyệt đối KHÔNG chèn câu giao tiếp. Chỉ trả về kết quả dịch.
                    3. Bỏ qua phần phụ lục tiếng Anh ở cuối tài liệu (nếu có).
                    4. Dịch chuẩn theo bộ từ điển:
                    {glossary}
                    5. TÁI TẠO BẢNG BIỂU: Nếu phát hiện bảng dữ liệu bị xô lệch cột do lỗi trích xuất (đặc biệt là các bảng có ô gộp), hãy tự động phân tích ngữ cảnh, tách hàng/cột hợp lý và vẽ lại thành một bảng Markdown hoàn chỉnh, thẳng hàng.
                    
                    Văn bản:
                    {chunk}
                    """
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    translated_text += response.text.strip() + "\n\n"
                    progress_bar.progress((i + 1) / len(chunks))
                
            st.success("Hoàn tất dịch thuật!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Bản gốc")
                st.text_area("Original Text", value=md_text, height=600, label_visibility="collapsed")
            with col2:
                st.subheader("Bản dịch")
                st.markdown(translated_text)
                
            zip_filename = "BanDich_KhoaHoc.zip"
            with zipfile.ZipFile(zip_filename, "w") as zipf:
                zipf.writestr("BanDich.md", translated_text)
                for root, _, files in os.walk(image_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=os.path.join("images", file))
                        
            with open(zip_filename, "rb") as f:
                st.download_button(
                    label="Tải File ZIP (Markdown + Ảnh)", 
                    data=f, 
                    file_name=zip_filename, 
                    mime="application/zip",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
else:
    st.info("Nhập API Key và tải file PDF để bắt đầu.")