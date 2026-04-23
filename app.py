import streamlit as st
from google import genai
from openai import OpenAI
import tempfile
import base64
import os
import cv2
import io
from PIL import Image

# ==========================================
# 1. 语言字典 (i18n)
# ==========================================
LANG = {
    "CN": {
        "title": "🏠 智能户型设计助手 (旗舰版)",
        "sidebar_title": "⚙️ 系统设置",
        "engine_label": "🤖 选择大脑引擎 (文本分析)",
        "gemini_key": "Gemini API Key (海外节点)",
        "doubao_key": "豆包 API Key (火山引擎通用)",
        "doubao_ep": "豆包 大脑接入点 (Vision Pro EP)",
        "doubao_draw_ep": "豆包 画笔接入点 (Seedream EP)",
        "step1": "📍 步骤 1: 上传户型图 (必填项)",
        "step2": "🏢 步骤 2: 样板间/现场实拍",
        "step2_desc": "支持图片。若上传视频，AI将自动提取关键画面。",
        "step3": "🎨 步骤 3: 期望风格参考 (选填项)",
        "step3_desc": "上传您真正喜欢的装修风格参考。",
        "step4": "💬 步骤 4: 具体改造需求",
        "req_placeholder": "例如：想要现代极简风格，主卧需要衣帽间...",
        "btn_generate": "🪄 开始智能设计",
        "btn_draw": "🎨 渲染 3D 效果图",
        "warning_api": "请在侧边栏完善必要的 API 密钥信息！",
        "warning_file": "请至少上传户型图和样板间参考！",
        "info_extracting": "🎥 正在自动提取并压缩视频关键帧...",
        "status_analyzing": "AI 大脑正在解析空间结构，生成方案中...",
        "status_drawing": "AI 画笔正在绘制空间效果图 (约 15 秒)...",
        "success": "✨ 方案设计完成！",
        "success_draw": "🖼️ 效果图渲染完毕！"
    },
    "EN": {
        "title": "🏠 AI Home Designer (Ultimate)",
        "sidebar_title": "⚙️ Settings",
        "engine_label": "🤖 Select Brain Engine",
        "gemini_key": "Gemini API Key",
        "doubao_key": "Doubao API Key",
        "doubao_ep": "Doubao Brain Endpoint (Vision Pro)",
        "doubao_draw_ep": "Doubao Drawing Endpoint (Seedream)",
        "step1": "📍 Step 1: Floor Plan (Required)",
        "step2": "🏢 Step 2: Showhouse / Reality",
        "step2_desc": "Images/Videos. Auto key-frame extraction.",
        "step3": "🎨 Step 3: Desired Style (Optional)",
        "step3_desc": "Upload a reference for the desired style.",
        "step4": "💬 Step 4: Specific Requirements",
        "req_placeholder": "e.g., Modern minimalist, walk-in closet...",
        "btn_generate": "🪄 Generate Design",
        "btn_draw": "🎨 Render 3D Image",
        "warning_api": "Please configure the API keys in the sidebar!",
        "warning_file": "Please upload at least a floor plan and showhouse!",
        "info_extracting": "🎥 Auto-extracting and compressing key frames...",
        "status_analyzing": "Analyzing the space... Please wait...",
        "status_drawing": "Rendering 3D image (approx. 15s)...",
        "success": "✨ Design Plan Completed!",
        "success_draw": "🖼️ Rendering Completed!"
    }
}

# ==========================================
# 2. 核心提示词与辅助函数 (加入智能压缩)
# ==========================================
def get_system_prompt(language, user_requirements, has_style_ref):
    style_logic = "严格参考【期望风格图】的色调和材质。" if has_style_ref else f"完全根据需求：【{user_requirements}】来构思。"
    if language == "CN":
        return f"你是一名顶尖全栈室内设计师。\n1. 【户型图】：看结构。\n2. 【实勘图/视频帧】：看真实层高采光，忽略老旧风格。\n3. 【期望风格】：{style_logic}\n\n请输出：\n### 📍 空间分析\n### 🛋️ 定制方案\n### 🎨 AI 提示词 (英文Midjourney格式)"
    else:
        return f"You are a top interior designer.\nAnalyze structure from Floor Plan and reality from Showhouse images/frames.\nStyle: {style_logic}\n\nOutput:\n### 📍 Spatial Analysis\n### 🛋️ Custom Design Plan\n### 🎨 AI Rendering Prompts"

def get_base64_image(uploaded_file):
    """处理静态图片转 Base64（带自动等比例压缩，防请求超载）"""
    if uploaded_file and uploaded_file.name.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
        try:
            img = Image.open(uploaded_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # 限制最大尺寸为 1024x1024，保持比例
            img.thumbnail((1024, 1024))
            buffered = io.BytesIO()
            # 压缩为 85% 质量的 JPEG
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            st.error(f"图片压缩失败: {e}")
            return None
    return None

def extract_frames_from_video_base64(video_file, num_frames=6):
    """从视频中均匀提取关键帧，自动缩放并转化为 Base64"""
    base64_frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}") as tmp:
        tmp.write(video_file.getvalue())
        tmp_path = tmp.name
        
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened(): return base64_frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        step = max(1, total_frames // num_frames)
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total_frames - 1))
            ret, frame = cap.read()
            if ret:
                # 动态缩放视频帧到最大 1024
                h, w = frame.shape[:2]
                scale = min(1024/w, 1024/h)
                if scale < 1:
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                # 压缩为 85% 质量的 JPEG
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
                
    cap.release()
    try: os.remove(tmp_path)
    except: pass
    return base64_frames

def append_media_to_doubao(content_list, uploaded_file, lang_code):
    if not uploaded_file: return
    ext = uploaded_file.name.lower().split('.')[-1]
    if ext in ['mp4', 'mov', 'avi']:
        st.toast(LANG[lang_code]["info_extracting"])
        for b64 in extract_frames_from_video_base64(uploaded_file, 6):
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    else:
        b64 = get_base64_image(uploaded_file)
        if b64: content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

# ==========================================
# 3. Streamlit 界面及状态初始化
# ==========================================
st.set_page_config(page_title="AI Home Designer", layout="wide")

# 初始化 session_state，用于保存文字分析结果，避免点击绘图按钮时文字消失
if "design_result" not in st.session_state:
    st.session_state.design_result = None
if "user_req_cache" not in st.session_state:
    st.session_state.user_req_cache = ""

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
    doubao_ep = st.sidebar.text_input(t["doubao_ep"], placeholder="ep-... (Vision Pro)")

st.sidebar.markdown("---")
# 绘图始终使用豆包的 Seedream 模型
doubao_draw_ep = st.sidebar.text_input(t["doubao_draw_ep"], placeholder="ep-... (Seedream)", help="用于渲染最终效果图")

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
# 4. 大脑处理逻辑 (文本生成)
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
                st.session_state.user_req_cache = user_req # 缓存用户需求供绘图使用
                
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
                    
                    response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
                    st.session_state.design_result = response.text

                elif engine_choice == "字节豆包 (Doubao)":
                    # 加入了 timeout=60.0 防卡死机制
                    client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=60.0)
                    content_list = [{"type": "text", "text": prompt_text}]
                    
                    st.toast("正在压缩并处理图片/视频...", icon="⏳")
                    append_media_to_doubao(content_list, floor_plan_file, lang_code)
                    append_media_to_doubao(content_list, showhouse_file, lang_code)
                    if has_style: append_media_to_doubao(content_list, style_file, lang_code)
                    st.toast("媒体处理完成，正在请求豆包大脑...", icon="🚀")
                    
                    response = client.chat.completions.create(model=doubao_ep, messages=[{"role": "user", "content": content_list}])
                    st.session_state.design_result = response.choices[0].message.content

                st.success(t["success"])

            except Exception as e:
                st.error(f"分析错误: {e}")

# ==========================================
# 5. 画笔处理逻辑 (图像渲染 - 仅在有文字方案后显示)
# ==========================================
if st.session_state.design_result:
    st.markdown("---")
    st.markdown(st.session_state.design_result)
    
    st.markdown("---")
    st.subheader("🖼️ AI 空间效果图渲染")
    
    # 点击渲染按钮
    if st.button(t["btn_draw"], type="primary", use_container_width=True):
        # 即使是大脑选了Gemini，绘图也需要豆包的 API Key 和 Seedream 接入点
        draw_key = api_key if engine_choice == "字节豆包 (Doubao)" else st.sidebar.text_input("必须输入豆包 API Key 进行绘图", type="password")
        
        if not draw_key or not doubao_draw_ep:
            st.error("⚠️ 请在左侧栏配置完整的 【豆包 API Key】 和 【豆包 画笔接入点 (Seedream EP)】")
        else:
            with st.spinner(t["status_drawing"]):
                try:
                    # 加入了 timeout=60.0 防卡死机制
                    draw_client = OpenAI(api_key=draw_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=60.0)
                    
                    # 组装参考图 (优先用样板间作为底图，再加风格图参考)
                    ref_images = []
                    sh_b64 = get_base64_image(showhouse_file)
                    if sh_b64: ref_images.append(f"data:image/jpeg;base64,{sh_b64}")
                    style_b64 = get_base64_image(style_file)
                    if style_b64: ref_images.append(f"data:image/jpeg;base64,{style_b64}")
                    
                    # 组装强力绘图 Prompt
                    draw_prompt = f"Interior design, highly detailed, photorealistic, 8k, Unreal Engine 5 render, cinematic lighting. Based on user requirements: {st.session_state.user_req_cache}"
                    
                    # 发起绘图请求
                    imagesResponse = draw_client.images.generate(
                        model=doubao_draw_ep, 
                        prompt=draw_prompt,
                        size="2K",
                        response_format="b64_json",
                        stream=True,
                        extra_body={
                            "image": ref_images if ref_images else None,
                            "watermark": False,
                            "sequential_image_generation": "auto",
                            "sequential_image_generation_options": {"max_images": 1}
                        }
                    )
                    
                    # 解析流式图片流
                    image_placeholder = st.empty()
                    for event in imagesResponse:
                        if event is None: continue
                        if event.type in ["image_generation.partial_succeeded", "image_generation.succeeded"]:
                            if event.b64_json:
                                image_data = base64.b64decode(event.b64_json)
                                image_placeholder.image(image_data, caption=t["success_draw"], use_container_width=True)
                                
                    st.toast(t["success_draw"])
                    
                except Exception as e:
                    st.error(f"渲染失败: {e}")
