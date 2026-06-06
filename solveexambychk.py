import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
from PIL import Image
import io
import time
import markdown
import re
import os
import pickle

# --- 1. การตั้งค่าเริ่มต้น และ Session State ---
keys_to_init = {
    'processed_data': {}, 
    'page_status': {}, # pending, generated, rechecked, review_1_done, review_2_done
    'page_idx': 0,
    'is_running': False,
    'stop_clicked': False,
    'exhausted_models': {},
    'current_active_model': "gemini-3.1-flash-lite",
    'flash_models_list': [], 
    'pdf_bytes': None,
    'pdf_name': "",
    'selected_pages': [],
    'show_reset_confirm': False,
    'estimated_tokens_used': 0,
    'user_api_key': "",
    'use_custom_prompt': False,
    'custom_prompt_text': "",
    'max_tokens_per_q': 3000
}

for k, v in keys_to_init.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.set_page_config(page_title="Exam Solver Space 📝", layout="wide")

# --- 2. Custom CSS & Minimalist Design ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Sarabun', sans-serif !important; }
    .stApp { background-color: #F8FAFC; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 1rem; letter-spacing: -0.5px; }
    
    /* Edit Box แบบ Minimalist */
    .edit-box { 
        border: 1px solid #E2E8F0; 
        border-radius: 12px; 
        padding: 24px; 
        background: #FFFFFF; 
        font-size: 16px; 
        line-height: 1.6;
        color: #1E293B;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05);
        height: 600px;
        overflow-y: auto;
    }
    
    /* การเน้นคำสำคัญ (สีแดงตาม Request) */
    strong, b { color: #DC2626; font-weight: 700; }
    
    /* Status Colors */
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid; }
    .status-gen { background: #EBF8FF; border-left-color: #3182CE; color: #2B6CB0; }
    .status-recheck { background: #FFFFF0; border-left-color: #D69E2E; color: #B7791F; }
    .status-rev1 { background: #F0FFF4; border-left-color: #38A169; color: #2F855A; }
    .status-rev2 { background: #FAF5FF; border-left-color: #805AD5; color: #553C9A; }
</style>
""", unsafe_allow_html=True)

# --- 3. ฟังก์ชัน Helper (Save/Load/Cleaning/Tagging) ---
def save_workspace():
    data = {k: st.session_state[k] for k in ['pdf_bytes', 'pdf_name', 'processed_data', 'page_status', 'selected_pages', 'user_api_key', 'custom_prompt_text', 'use_custom_prompt', 'max_tokens_per_q', 'current_active_model']}
    try:
        with open("exam_workspace.pkl", "wb") as f: pickle.dump(data, f)
    except: pass

def load_workspace():
    if os.path.exists("exam_workspace.pkl"):
        try:
            with open("exam_workspace.pkl", "rb") as f:
                data = pickle.load(f)
                for k, v in data.items(): st.session_state[k] = v
            return True
        except: return False
    return False

def calc_dynamic_fontsize(text, rect_width, rect_height):
    if not text or rect_width <= 0 or rect_height <= 0: return 16
    area = rect_width * rect_height
    char_count = len(text)
    if char_count == 0: return 16
    estimated_size = (area / (char_count * 0.45)) ** 0.5
    return max(12, min(32, int(estimated_size))) 

def clean_ai_response(text):
    # ฟังก์ชันลบคำเกริ่นนำและข้อความแถมของ AI
    match = re.search(r'(Q\s*\d+\s*:)', text)
    if match:
        text = text[match.start():]
        
    # ลบข้อความจบท้ายเชิงสนทนา และคำศัพท์ของระบบ
    text = re.sub(r'(?i)(จากการตรวจสอบ|พบว่าเนื้อหา|ครบถ้วนตาม|เป็นไปตามโจทย์|ดังนี้|ครับ|ค่ะ|ALL_COMPLETE|NO_MISSING).*?$', '', text, flags=re.MULTILINE)
    
    # บังคับลบช่องว่างบรรทัดที่ว่างเปล่า
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def apply_exam_tags(text):
    # 1. บังคับลบช่องว่างระหว่างข้อ (ยุบให้เหลือแค่บรรทัดใหม่บรรทัดเดียว)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 2. ป้องกันบั๊กที่ AI ชอบใส่ ** ครอบ Prefix ของเรา
    text = re.sub(r'\*\*(Q\s*\d+\s*:)\*\*', r'\1', text)
    text = re.sub(r'\*\*(A\s*:)\*\*', r'\1', text)
    text = re.sub(r'\*\*(HY\s*:)\*\*', r'\1', text)
    text = re.sub(r'\*\*(TRICK\s*:)\*\*', r'\1', text)
    
    # 3. เน้นคำสำคัญที่ AI ใส่ ** ** ให้กลายเป็นตัวหนาสีแดง
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #DC2626; font-weight: bold;">\1</strong>', text)
    
    # 4. แทรกเส้นคั่นบางๆ ก่อนขึ้นข้อใหม่ (Q) ที่ไม่ได้อยู่บรรทัดแรกสุด
    # โดยนำ <hr> ไปแทนที่ \n เพื่อให้เป็นเส้นแบ่งที่ไม่กินพื้นที่บรรทัด
    text = re.sub(r'\n(Q\s*\d+\s*:)', r'<hr style="border: 0; border-top: 1px dashed #CBD5E1; margin: 8px 0; padding: 0;">\1', text)
    
    # 5. เติม space 2 ตัวหน้า \n ที่เหลือ เพื่อบังคับให้ Markdown แปลงเป็น Line Break (<br>) อย่างถูกต้อง
    text = text.replace('\n', '  \n')
    
    # 6. แปลง Markdown เป็น HTML พื้นฐาน
    html = markdown.markdown(text, extensions=['tables'])
    
    # 7. สีหัวข้อต่างๆ
    html = re.sub(r'(Q\s*\d+\s*:)', r'<span style="color: #0369A1; font-weight: bold;">\1</span>', html)
    html = re.sub(r'(A\s*:)', r'<span style="color: #15803D; font-weight: bold;">\1</span>', html)
    html = re.sub(r'(HY\s*:)', r'<span style="color: #BE123C; font-weight: bold;">\1</span>', html)
    html = re.sub(r'(TRICK\s*:)', r'<span style="color: #EAB308; font-weight: bold;">\1</span>', html)
    
    # 8. ลบ Tag <p> ที่เว้นช่องว่างเกินความจำเป็น และจัดระเบียบ <br>
    html = html.replace('<p>', '').replace('</p>', '<br>')
    html = re.sub(r'(<br>\s*){2,}', '<br>', html) # ยุบ <br> ที่ซ้ำกัน
    
    # 9. ลบ <br> ด้านหน้าและหลังสุดทิ้งเพื่อความสวยงาม
    html = re.sub(r'^(<br>|\s)+', '', html)
    html = re.sub(r'(<br>|\s)+$', '', html)
        
    return html

# --- 4. Sidebar: Settings ---
with st.sidebar:
    st.markdown("## ⚙️ Settings (ตั้งค่า)")
    api_key = st.text_input("🔑 Gemini API Key:", type="password", value=st.session_state.user_api_key)
    if api_key != st.session_state.user_api_key:
        st.session_state.user_api_key = api_key
        save_workspace()
        st.rerun()
        
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            flash_models = [m for m in models if "flash" in m.lower() or "3.1" in m.lower()]
            flash_models = sorted(flash_models, key=lambda x: 0 if "3.1-flash-lite" in x else 1)
            st.session_state.flash_models_list = flash_models
            
            selected_model = st.selectbox("AI Model:", flash_models, index=0 if st.session_state.current_active_model not in flash_models else flash_models.index(st.session_state.current_active_model))
            st.session_state.current_active_model = selected_model
            st.success("✅ API พร้อมใช้งาน")
        except: 
            st.error("❌ API Key ไม่ถูกต้อง")

    st.markdown("### 🛠️ การทำงาน")
    st.session_state.max_tokens_per_q = st.number_input("Max Tokens ต่อข้อ", min_value=500, max_value=8000, value=st.session_state.max_tokens_per_q, step=500)
    
    st.markdown("### 📐 Layout PDF ขาออก")
    right_margin_pct = st.slider("เพิ่มพื้นที่ด้านขวาสำหรับเขียนเฉลย (%)", 50, 150, 100, help="100% คือเพิ่มพื้นที่กระดาษทางขวาให้ใหญ่เท่ากระดาษต้นฉบับ")
    
    st.markdown("### 🤖 Prompt อัจฉริยะ")
    use_custom = st.checkbox("แก้ไข Prompt หลัก", value=st.session_state.use_custom_prompt)
    
    # อัปเดต Prompt ใหม่ ให้จัดการข้อสอบแหว่งและรอยเฉลยเดิม
    base_prompt = f"""คุณคืออาจารย์แพทย์ที่กำลังทำเฉลยข้อสอบ
หน้าที่ของคุณคืออ่านรูปหน้าหลัก หาว่ามีโจทย์ทั้งหมดกี่ข้อ แล้วทำเฉลยเรียงข้อกันลงมา
(ให้ดูภาพหน้าถัดไปประกอบด้วย เฉพาะในกรณีที่ข้อสุดท้ายของหน้าหลักมีตัวเลือกหรือเนื้อหาล้นไปหน้าถัดไป)

**กฎเหล็ก (ห้ามฝ่าฝืน):**
1. **เฉลยเฉพาะข้อที่มีจุดเริ่มต้นในภาพหน้าหลักเท่านั้น** ห้ามเฉลยเนื้อหาของหน้าอื่นเด็ดขาด!
2. **ต้องระบุเลขข้อให้ตรงกับในรูปเป๊ะๆ** เช่น ในรูปคือข้อ 27 ต้องใช้ "Q27:" ห้ามรันเลข 1,2,3 เอง
3. **ห้ามพิมพ์ข้อความสนทนาเกริ่นนำหรือสรุปท้ายใดๆ ทั้งสิ้น** ให้เริ่มที่ Q[เลขข้อ]: ทันที
4. **กรณีตัวเลือกมาไม่ครบ:** หากในรูปตัวเลือกมาไม่ครบ และไม่มีข้อที่ถูกในตัวเลือกที่มีอยู่ ให้ระบุแจ้งด้วยว่า "ไม่มีข้อถูกในตัวเลือกที่มี" พร้อมบอกคำตอบที่แท้จริง
5. **กรณีรูปมีเฉลยเดิมอยู่แล้ว:** หากในรูปมีรอยกากบาท, ไฮไลท์ หรือเขียนเฉลยมาแล้ว ให้คุณ "ตรวจทานรอยนั้น" ว่าถูกหรือไม่ ถ้าผิดให้อธิบายว่าทำไมรอยนั้นผิด และแก้เป็นข้อที่ถูก
6. ถ้าโจทย์ไม่สมบูรณ์ให้ "เดาใจ" ว่าต้องการถามอะไร และการอธิบายใน A: ต้องละเอียด ชัดเจน **มีเหตุผลโดยใช้คำว่า '...เพราะ...' เสมอ**
7. **เน้นคำสำคัญ** โดยใส่เครื่องหมาย **ครอบคำนั้น (เช่น **Fever**)
8. **ห้ามเว้นบรรทัดว่างระหว่างข้อ** ให้ Q ของข้อถัดไปต่อท้าย TRICK ของข้อก่อนหน้าในบรรทัดใหม่ทันที
9. ห้ามเฉลยเกิน {st.session_state.max_tokens_per_q} คำต่อ 1 ข้อ

**รูปแบบการตอบ (ต้องเป๊ะตามนี้ ห้ามเว้นบรรทัดว่างคั่นระหว่างข้อ):**
Q[เลขข้อจริง]: [โจทย์ถามอะไร?]
A: [ตอบอะไร? (และชี้แจงรอยเฉลยเดิมถ้ามี) ทำไมข้อนี้ถูกและข้ออื่นผิด อธิบายอย่างละเอียด]
HY: [สรุป High-Yield สั้นๆกระชับ]
TRICK: [ทริคการจำ]
"""
    if use_custom:
        custom_prompt = st.text_area("แก้ไข Prompt", value=st.session_state.custom_prompt_text or base_prompt, height=450)
        st.session_state.custom_prompt_text = custom_prompt
    else:
        st.session_state.custom_prompt_text = base_prompt
    st.session_state.use_custom_prompt = use_custom

# --- 5. Main Content: Upload & Processing ---
st.markdown("<div class='main-header'>📝 Exam Note Space: 4-Phase Solver</div>", unsafe_allow_html=True)

if not st.session_state.pdf_bytes:
    if os.path.exists("exam_workspace.pkl"):
        if st.button("🔄 กู้คืนงานเฉลยข้อสอบที่ทำค้างไว้", type="primary"):
            if load_workspace(): st.rerun()
            
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ข้อสอบ (PDF)", type="pdf")
    if uploaded_file:
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        st.session_state.pdf_name = uploaded_file.name
        doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
        st.session_state.selected_pages = list(range(len(doc)))
        for i in range(len(doc)): st.session_state.page_status[i] = 'pending'
        save_workspace()
        st.rerun()
else:
    doc_in = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
    total_pages = len(doc_in)
    
    st.info(f"📄 ไฟล์: **{st.session_state.pdf_name}** ({total_pages} หน้า)")
    
    c1, c2, c3 = st.columns([1,1,2])
    if c1.button("🚀 เริ่ม / ทำต่อ (Start)", type="primary"):
        st.session_state.is_running = True
        st.session_state.stop_clicked = False
    if c2.button("🛑 หยุดชั่วคราว (Stop)"):
        st.session_state.is_running = False
        st.session_state.stop_clicked = True
    if c3.button("🗑️ ล้างข้อมูลและอัปโหลดใหม่"):
        if os.path.exists("exam_workspace.pkl"): os.remove("exam_workspace.pkl")
        for k in keys_to_init: del st.session_state[k]
        st.rerun()

    action_placeholder = st.empty()

    # --- 6. E-Book Reader & Editor ---
    st.write("---")
    curr = st.session_state.page_idx
    
    col_v1, col_v2 = st.columns([1, 1.2])
    with col_v1:
        st.subheader(f"📖 ข้อสอบหน้า {curr + 1}")
        cn1, cn2, cn3 = st.columns([1,2,1])
        if cn1.button("⬅️ ก่อนหน้า") and curr > 0: st.session_state.page_idx -= 1; st.rerun()
        if cn3.button("ถัดไป ➡️") and curr < total_pages - 1: st.session_state.page_idx += 1; st.rerun()
        
        p_img = doc_in[curr].get_pixmap(dpi=100)
        st.image(p_img.tobytes("png"), use_container_width=True)

    with col_v2:
        st.subheader(f"💡 เฉลย AI (สถานะ: {st.session_state.page_status.get(curr, 'pending')})")
        if curr in st.session_state.processed_data:
            data = st.session_state.processed_data[curr]
            if f"edit_{curr}" not in st.session_state: st.session_state[f"edit_{curr}"] = False
            
            raw_display = data["user_text"] if data["user_text"] else data["ai_text"]
            clean_display = clean_ai_response(raw_display)
            
            if st.session_state[f"edit_{curr}"]:
                edited = st.text_area("แก้ไขเฉลย:", value=clean_display, height=500)
                ce1, ce2 = st.columns(2)
                if ce1.button("💾 บันทึกการแก้ไข", type="primary"):
                    st.session_state.processed_data[curr]["user_text"] = edited
                    st.session_state[f"edit_{curr}"] = False
                    save_workspace(); st.rerun()
                if ce2.button("ยกเลิก"):
                    st.session_state[f"edit_{curr}"] = False; st.rerun()
            else:
                html_content = apply_exam_tags(clean_display)
                st.markdown(f"<div class='edit-box'>{html_content}</div>", unsafe_allow_html=True)
                if st.button("✏️ แก้ไขเฉลยหน้านี้ (ล็อคข้อมูล)"):
                    st.session_state[f"edit_{curr}"] = True
                    st.session_state.is_running = False 
                    st.rerun()
        else:
            st.info("⏳ รอการประมวลผล...")

    # --- 7. Download Final PDF ---
    st.write("---")
    if len(st.session_state.processed_data) > 0:
        if st.button("📦 ดาวน์โหลด PDF ฉบับสมบูรณ์"):
            with st.spinner("กำลังประกอบร่างเนื้อหาเฉลยลงใน PDF..."):
                doc_out = fitz.open()
                arch = fitz.Archive(".")
                
                for i in range(total_pages):
                    p_in = doc_in[i]; w, h = p_in.rect.width, p_in.rect.height
                    new_w = w * (1 + right_margin_pct/100)
                    p_out = doc_out.new_page(width=new_w, height=h)
                    p_out.show_pdf_page(fitz.Rect(0, 0, w, h), doc_in, i)
                    
                    ans_box = fitz.Rect(w, 0, new_w, h)
                    p_out.draw_rect(ans_box, color=(0.99, 0.99, 0.99), fill=(0.99, 0.99, 0.99), width=0)
                    
                    if i in st.session_state.processed_data:
                        raw_txt = st.session_state.processed_data[i]["user_text"] or st.session_state.processed_data[i]["ai_text"]
                        if raw_txt:
                            clean_txt = clean_ai_response(raw_txt)
                            text_rect = fitz.Rect(w + 20, 20, new_w - 20, h - 20)
                            f_size = calc_dynamic_fontsize(clean_txt, text_rect.width, text_rect.height)
                            html = apply_exam_tags(clean_txt)
                            
                            # เพิ่ม hr CSS สำหรับเส้นประที่แบ่งข้อ
                            css = f"""
                            @font-face {{ font-family: 'T'; src: url('THSarabunNew.ttf'); }}
                            @font-face {{ font-family: 'T'; font-weight: bold; src: url('THSarabunNew Bold.ttf'); }}
                            body {{ font-family: 'T'; font-size: {f_size}px; line-height: 1.2; color: #1E293B; margin: 0; padding: 0; }} 
                            b, strong {{ font-weight: bold; }} 
                            hr {{ border: 0; border-top: 1px dashed #CBD5E1; margin: 4px 0; padding: 0; }}
                            """
                            try:
                                p_out.insert_htmlbox(text_rect, f"<style>{css}</style><body>{html}</body>", archive=arch)
                            except:
                                p_out.insert_textbox(text_rect, clean_txt, fontsize=f_size)
                                
                pdf_res = doc_out.tobytes()
                st.download_button("💾 ดาวน์โหลดเฉลยข้อสอบ (PDF)", data=pdf_res, file_name=f"Solved_{st.session_state.pdf_name}", mime="application/pdf")

    # --- 8. Background 4-Phase AI Processing ---
    if st.session_state.is_running and not st.session_state.stop_clicked:
        target_page = None
        current_phase = None
        
        target_page = next((i for i in st.session_state.selected_pages if st.session_state.page_status.get(i) == 'pending'), None)
        if target_page is not None: current_phase = 'generate'
        
        if target_page is None:
            target_page = next((i for i in st.session_state.selected_pages if st.session_state.page_status.get(i) == 'generated'), None)
            if target_page is not None: current_phase = 'recheck'
            
        if target_page is None:
            target_page = next((i for i in st.session_state.selected_pages if st.session_state.page_status.get(i) == 'rechecked'), None)
            if target_page is not None: current_phase = 'review_1'
            
        if target_page is None:
            target_page = next((i for i in st.session_state.selected_pages if st.session_state.page_status.get(i) == 'review_1_done'), None)
            if target_page is not None: current_phase = 'review_2'

        if target_page is not None:
            active_m = st.session_state.current_active_model
            model = genai.GenerativeModel(active_m)
            
            api_max_tokens = st.session_state.max_tokens_per_q * 10
            config = GenerationConfig(max_output_tokens=api_max_tokens)
            safety = { HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE }
            
            p_img = doc_in[target_page].get_pixmap(dpi=75)
            img_current = Image.open(io.BytesIO(p_img.tobytes("png")))
            
            payload = [st.session_state.custom_prompt_text if current_phase == 'generate' else "", "ภาพหน้าหลัก:", img_current]
            
            if target_page + 1 < total_pages:
                p_img_next = doc_in[target_page + 1].get_pixmap(dpi=75)
                img_next = Image.open(io.BytesIO(p_img_next.tobytes("png")))
                payload.extend(["ภาพหน้าถัดไป (ใช้ดูประกอบสำหรับข้อที่ล้นจากหน้าหลักเท่านั้น ห้ามเฉลยข้อที่เป็นของหน้านี้):", img_next])
            
            if current_phase == 'generate':
                action_placeholder.markdown(f"<div class='status-box status-gen'><b>[Phase 1] ⚙️ กำลังแกะโจทย์และเฉลย หน้า {target_page+1} (Cross-page mode)</b></div>", unsafe_allow_html=True)
                next_status = 'generated'
                
            elif current_phase == 'recheck':
                old_text = st.session_state.processed_data[target_page]["ai_text"]
                prompt = f"""นี่คือเฉลยที่คุณทำไว้สำหรับหน้านี้: \n\n{old_text}\n\n
                **คำสั่ง Recheck (ห้ามแก้ไขเนื้อหาเดิมเด็ดขาด):** ให้เทียบกับรูปภาพอีกครั้ง ว่าคุณเฉลย "ครบทุกข้อที่อยู่ในรูปหน้าหลัก" หรือยัง? 
                - หากมีข้อสอบตกหล่น: ให้เขียน **เฉพาะข้อที่ตกหล่นเพิ่มเติม** ตามฟอร์แมต Q[เลขข้อ]:, A:, HY:, TRICK:
                - หากเนื้อหาเดิมขาดหายไม่สมบูรณ์: ให้เขียนต่อให้จบ
                - หากเฉลยครบถ้วนทุกข้อในหน้าหลักแล้ว: ให้พิมพ์คำว่า "ALL_COMPLETE" คำเดียวเท่านั้น ห้ามพิมพ์อย่างอื่นเด็ดขาด!"""
                payload[0] = prompt
                action_placeholder.markdown(f"<div class='status-box status-recheck'><b>[Phase 2] 🔍 ตรวจสอบหาข้อตกหล่น หน้า {target_page+1}</b></div>", unsafe_allow_html=True)
                next_status = 'rechecked'
                
            elif current_phase == 'review_1':
                action_placeholder.markdown(f"<div class='status-box status-rev1'><b>[Phase 3] 📝 Review 1 จัดระเบียบข้อความ หน้า {target_page+1}</b></div>", unsafe_allow_html=True)
                next_status = 'review_1_done'
                
            elif current_phase == 'review_2':
                action_placeholder.markdown(f"<div class='status-box status-rev2'><b>[Phase 4] ✨ Review 2 Final Polish หน้า {target_page+1}</b></div>", unsafe_allow_html=True)
                next_status = 'review_2_done'

            try:
                if current_phase in ['review_1', 'review_2']:
                    final_text = clean_ai_response(st.session_state.processed_data[target_page]["ai_text"])
                    st.session_state.processed_data[target_page]["ai_text"] = final_text
                    st.session_state.page_status[target_page] = next_status
                    save_workspace()
                else:
                    resp = model.generate_content(payload, safety_settings=safety, generation_config=config)
                    new_text = resp.text.strip()
                    
                    if current_phase == 'recheck':
                        if "ALL_COMPLETE" in new_text or "ครบถ้วน" in new_text or "ตกหล่น" in new_text and len(new_text) < 100:
                            final_text = st.session_state.processed_data[target_page]["ai_text"]
                        else:
                            final_text = st.session_state.processed_data[target_page]["ai_text"] + "\n" + new_text
                    else:
                        final_text = new_text
                        
                    final_text = clean_ai_response(final_text)
                    
                    st.session_state.processed_data[target_page] = {"ai_text": final_text, "user_text": st.session_state.processed_data.get(target_page, {}).get("user_text", ""), "img": p_img.tobytes("png")}
                    st.session_state.page_status[target_page] = next_status
                    save_workspace()
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    action_placeholder.warning("⚠️ โควต้าเต็ม รอสักครู่...")
                    time.sleep(5)
                else:
                    st.session_state.processed_data[target_page] = {"ai_text": f"⚠️ Error [{current_phase}]: {error_msg}", "user_text": "", "img": p_img.tobytes("png")}
                    st.session_state.page_status[target_page] = next_status
            
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.is_running = False
            action_placeholder.success("✅ เสร็จสิ้นกระบวนการ 4-Phase ครบทุกหน้าแล้ว! (คลีนข้อความสนทนาเรียบร้อย) กดดาวน์โหลด PDF ได้เลยครับ")
