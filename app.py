import streamlit as st
from google import genai
from openai import OpenAI
import tempfile
import base64
import os
import cv2
from PIL import Image

# ==========================================
# 1. 语言字典 (i18n)
# ==========================================
LANG = {
    "CN": {
        "title": "🏠 智能户型设计助手 (全球双引擎版)",
        "sidebar_title": "⚙️ 系统设置",
        "engine_label": "🤖 选择 AI 引擎",
        "gemini_key": "Gemini API Key (海外节点)",
        "doubao_key": "豆包 API Key (国内节点)",
        "doubao_ep": "豆包 推理接入点 (Endpoint ID)",
        "step1": "📍 步骤 1: 上传户型图 (必填项)",
        "step2": "🏢 步骤 2: 样板间/现场实拍",
        "step2_desc": "支持图片。若上传视频，AI将自动提取关键画面分析。",
        "step3": "🎨 步骤 3: 期望风格参考 (选填项)",
        "step3_desc": "上传您真正喜欢的装修风格参考。",
        "step4": "💬 步骤 4: 具体改造需求",
        "req_placeholder": "例如：想要现代极简风格，主卧需要衣帽间...",
        "btn_generate": "🪄 开始智能设计",
        "warning_api": "请在侧边栏完善当前引擎的 API 密钥信息！",
        "warning_file": "请至少上传户型图和样板间参考！",
        "info_extracting": "🎥 正在从视频中自动提取 6 张关键帧用于空间分析...",
        "status_analyzing": "AI 引擎正在全速解析空间结构，生成方案中...",
        "success": "✨ 设计完成！"
    },
    "EN": {
        "title": "🏠 AI Home Designer (Dual Engine)",
        "sidebar_title": "⚙️ Settings",
        "engine_label": "🤖 Select AI Engine",
        "gemini_key": "Gemini API Key (Global)",
        "doubao_key": "Doubao API Key (China)",
        "doubao_ep": "Doubao Endpoint ID",
        "step1": "📍 Step 1: Floor Plan (Required)",
        "step2": "🏢 Step 2: Showhouse / Reality",
        "step2_desc": "Images/Videos. Auto key-frame extraction for videos.",
        "step3": "🎨 Step 3: Desired Style (Optional)",
        "step3_desc": "Upload a reference for the desired style.",
        "step4": "💬 Step 4: Specific Requirements",
        "req_placeholder": "e.g., Modern minimalist, walk-in closet...",
        "btn_generate": "🪄 Generate Design",
        "warning_api": "Please configure the API key for the selected engine!",
        "warning_file": "Please upload at least a floor plan and showhouse!",
        "info_extracting": "🎥 Auto-extracting 6 key frames from video...",
        "status_analyzing": "AI Engine is analyzing the space... Please wait...",
        "success": "✨ Design Completed!"
    }
}

# ==========================================
# 2. 核心提示词
# ==========================================
def get_system_prompt(language, user_requirements, has_style_ref):
    style_logic_cn = "我提供了【期望风格参考图】，请严格按照该参考的色调、材质和软装氛围进行设计。" if has_style_ref else "我没有提供具体的风格参考图，请完全根据我的【具体改造需求】文字描述来构思风格。"
    style_logic_en = "I have provided a [Desired Style Reference]. Please strictly follow its vibe." if has_style_ref else "I have not provided a style reference image. Please conceptualize based on my text."

    if language == "CN":
        return f"你是一名顶尖全栈室内设计师。\n1. 【户型图】：理解硬性尺寸和格局。\n2. 【样板间实勘】：理解真实层高、采光，忽略其老旧风格。如果你看到多张相似视角的图片，那是从实拍视频中提取的关键帧组合。\n3. 【期望风格】：{style_logic_cn}\n\n客户需求：【{user_requirements}】\n\n请输出：\n### 📍 空间结构与实勘分析\n### 🛋️ 定制化设计方案\n### 🎨 AI 效果图提示词 (英文，Midjourney格式)"
    else:
        return f"You are a top interior designer.\n1. [Floor Plan]: Structure/layout.\n2. [Reality]: Lighting/volume. If you see multiple frames, they are keyframes extracted from a video.\n3. [Style]: {style_logic_en}\n\nNeeds: [{user_requirements}]\n\nOutput:\n### 📍 Spatial Analysis\n### 🛋️ Custom Design Plan\n### 🎨 AI Rendering Prompts (Midjourney/SD format)"

# ==========================================
# 3. 媒体处理辅助函数 (视频抽帧核心算法)
# ==========================================
def get_base64_image(uploaded_file):
    """处理静态图片转 Base64"""
    if uploaded_file:
        return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    return None

def extract_frames_from_video_base64(video_file, num_frames=6):
    """从上传的视频流中均匀提取关键帧，转化为 Base64 数组"""
    base64_frames = []
    
    # 视频处理必须先落盘为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}") as tmp:
        tmp.write(video_file.getvalue())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return base64_frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        # 计算等距采样的步长
        step = max(1, total_frames // num_frames)
        
        for i in range(num_frames):
            frame_id = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                # 将 BGR 帧编码为 JPEG，再转为 Base64
                _, buffer = cv2.imencode('.jpg', frame)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(b64_str)

    cap.release()
    try:
        os.remove(tmp_path) # 用完清理垃圾
    except:
        pass
        
    return base64_frames

def append_media_to_doubao(content_list, uploaded_file, lang_code):
    """智能判断文件类型：图片直接加，视频自动抽帧加"""
    if not uploaded_file:
        return
        
    ext = uploaded_file.name.lower().split('.')[-1]
    
    if ext in ['mp4', 'mov', 'avi']:
        st.toast(LANG[lang_code]["info_extracting"]) # 页面右下角弹出提示
        b64_frames = extract_frames_from_video_base64(uploaded_file, num_frames=6)
        for b64 in b64_frames:
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    else:
        b64 = get_base64_image(uploaded_file)
        if b64:
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

# ==========================================
# 4. Streamlit 界面逻辑
# ==========================================
st.set_page_config(page_title="AI Home Designer Dual", layout="wide")

st.sidebar.selectbox("Language / 语言", ["CN", "EN"], key="lang")
lang_code = st.session_state.lang
t = LANG[lang_code]

st.sidebar.header(t["sidebar_title"])

engine_choice = st.sidebar.radio(t["engine_label"], ["Google Gemini 2.5", "字节豆包 (Doubao)"])

api_key = ""
doubao_ep = ""
if engine_choice == "Google Gemini 2.5":
    api_key = st.sidebar.text_input(t["gemini_key"], type="password")
else:
    api_key = st.sidebar.text_input(t["doubao_key"], type="password")
    doubao_ep = st.sidebar.text_input(t["doubao_ep"], placeholder="ep-xxxxxx-xxxx")

st.title(t["title"])

col1, col2 = st.columns(2)
with col1:
    st.subheader(t["step1"])
    floor_plan_file = st.file_uploader("Floor Plan", type=["jpg", "png", "jpeg"], key="fp", label_visibility="collapsed")
    if floor_plan_file: st.image(floor_plan_file, use_container_width=True)

with col2:
    st.subheader(t["step2"])
    st.caption(t["step2_desc"])
    showhouse_file = st.file_uploader("Showhouse", type=["jpg", "png", "jpeg", "mp4", "gif", "mov"], key="sh", label_visibility="collapsed")
    if showhouse_file:
        if showhouse_file.name.lower().endswith(('mp4', 'mov')): st.video(showhouse_file)
        else: st.image(showhouse_file, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.subheader(t["step3"])
    st.caption(t["step3_desc"])
    style_file = st.file_uploader("Style", type=["jpg", "png", "jpeg", "mp4", "gif"], key="sf", label_visibility="collapsed")
    if style_file:
        if style_file.name.lower().endswith(('mp4', 'mov')): st.video(style_file)
        else: st.image(style_file, use_container_width=True)

with col4:
    st.subheader(t["step4"])
    user_req = st.text_area("Requirements", placeholder=t["req_placeholder"], height=150, label_visibility="collapsed")

# ==========================================
# 5. 核心处理逻辑 (双引擎路由)
# ==========================================
if st.button(t["btn_generate"], type="primary", use_container_width=True):
    if not api_key or (engine_choice == "字节豆包 (Doubao)" and not doubao_ep):
        st.error(t["warning_api"])
    elif not floor_plan_file or not showhouse_file:
        st.error(t["warning_file"])
    else:
        with st.spinner(t["status_analyzing"]):
            try:
                has_style = style_file is not None
                prompt_text = get_system_prompt(lang_code, user_req, has_style)
                
                # ------------------------------------
                # 路由 A：Gemini 原生视频解析
                # ------------------------------------
                if engine_choice == "Google Gemini 2.5":
                    client = genai.Client(api_key=api_key)
                    contents = [prompt_text]
                    
                    def process_gemini_file(f):
                        if f is None: return None
                        if f.name.lower().endswith(('mp4', 'mov', 'gif')):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as tmp:
                                tmp.write(f.getvalue())
                                return client.files.upload(file=tmp.name)
                        return Image.open(f)
                    
                    if floor_plan_file: contents.append(process_gemini_file(floor_plan_file))
                    if showhouse_file: contents.append(process_gemini_file(showhouse_file))
                    if style_file: contents.append(process_gemini_file(style_file))
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=contents
                    )
                    st.success(t["success"])
                    st.markdown(response.text)

                # ------------------------------------
                # 路由 B：豆包 智能抽帧解析
                # ------------------------------------
                elif engine_choice == "字节豆包 (Doubao)":
                    client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
                    
                    content_list = [{"type": "text", "text": prompt_text}]
                    
                    # 依次处理三个文件，遇到视频会自动拆成 6 张图塞进去
                    append_media_to_doubao(content_list, floor_plan_file, lang_code)
                    append_media_to_doubao(content_list, showhouse_file, lang_code)
                    if has_style:
                        append_media_to_doubao(content_list, style_file, lang_code)
                    
                    messages = [{"role": "user", "content": content_list}]
                    
                    response = client.chat.completions.create(
                        model=doubao_ep, 
                        messages=messages
                    )
                    st.success(t["success"])
                    st.markdown(response.choices[0].message.content)

            except Exception as e:
                st.error(f"发生错误 / Error: {e}")
