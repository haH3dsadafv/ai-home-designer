import streamlit as st
from google import genai
from openai import OpenAI
import tempfile
import base64
import os
import cv2
import io
import re
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
        "step2": "🏢 步骤 2: 上传样板间实勘 (MP4/图/GIF)",
        "step2_desc": "全屋参考。若视频过大将自动抽帧。AI将自动匹配房间画面。",
        "step3": "🎨 步骤 3: 期望风格参考 (选填项)",
        "step3_desc": "上传您真正喜欢的装修风格参考图。",
        "step4": "💬 步骤 4: 具体改造需求",
        "req_placeholder": "例如：现代中古，主卧需要衣帽间...",
        "btn_generate": "🪄 生成全屋分析与设计方案",
        "label_drawing_room": "选择要绘制效果图的房间:",
        "option_drawing_room": ["大横厅 (Living Room)", "主卧 (Master Bedroom)", "次卧 (Bedroom 2)", "厨房 (Kitchen)", "电梯厅 (Elevator Hall)"],
        "btn_draw": "🎨 智能渲染 3D 效果图",
        "warning_api": "请在侧边栏完善大脑和画笔的 API 密钥信息！",
        "warning_file": "请至少上传户型图和样板间参考！",
        "info_extracting": "🎥 正在自动提取视频/GIF关键帧用于分析...",
        "status_analyzing": "AI 大脑正在解析空间结构与尺寸，生成全屋方案中...",
        "status_drawing": "AI 画笔正在绘制 spatial-aware 效果图 (约 15 秒)...",
        "success": "✨ 全屋方案设计完成！",
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
        "step2": "🏢 Step 2: Showhouse Reality (MP4/IMG/GIF)",
        "step2_desc": "Global ref. AI will auto-match room frames from video.",
        "step3": "🎨 Step 3: Desired Style (Optional)",
        "step3_desc": "Upload a reference for the desired style.",
        "step4": "💬 Step 4: Specific Requirements",
        "req_placeholder": "e.g., Modern minimalist, walk-in closet...",
        "btn_generate": "🪄 Generate Global Analysis & Plan",
        "label_drawing_room": "Select Room to Render:",
        "option_drawing_room": ["Living Room", "Master Bedroom", "Bedroom 2", "Kitchen", "Elevator Hall"],
        "btn_draw": "🎨 Smart Render 3D Image",
        "warning_api": "Please configure Brain and Drawing API keys in the sidebar!",
        "warning_file": "Please upload at least a floor plan and showhouse!",
        "info_extracting": "🎥 Auto-extracting key frames...",
        "status_analyzing": "Analyzing the space & dimensions... Please wait...",
        "status_drawing": "Rendering 3D image (approx. 15s)...",
        "success": "✨ Global Design Plan Completed!",
        "success_draw": "🖼️ Rendering Completed!"
    }
}

# ==========================================
# 2. 核心提示词与辅助函数 (加入智能压缩与AI寻帧引擎)
# ==========================================
def get_system_prompt(language, user_requirements, has_style_ref):
    style_logic = "严格参考【期望风格图】的色调和材质。" if has_style_ref else f"完全根据需求：【{user_requirements}】来构思。"
    if language == "CN":
        return f"你是一名顶尖室内设计师。\n1. 【全局户型图】：看结构。**务必读取尺寸数值作为约束**。\n2. 【实勘视频/帧】：看真实层高采光。\n3. 【期望风格】：{style_logic}\n\n客户需求：【{user_requirements}】\n\n请输出：\n### 📍 空间尺寸与分割分析\n### 🛋️ 定制化全屋设计方案\n### 🎨 AI 效果图描述词 (英文 Midjourney格式)"
    else:
        return f"You are a top interior designer.\nAnalyze structure from Floor Plan & Showhouse. **Must read dimensions**.\nStyle: {style_logic}\nNeeds: [{user_requirements}]\n\nOutput:\n### 📍 Spatial Analysis\n### 🛋️ Custom Global Design Plan\n### 🎨 AI Rendering Prompts"

def get_drawing_prompt_desc(selected_room, plan_context, user_requirements, has_style_ref):
    return f"Highly detailed, photorealistic interior rendering of the {selected_room}, strict architectural integrity. Plan context: {plan_context[:200]}... User needs: {user_requirements}. Trending on ArtStation, 8k resolution, Unreal Engine 5 render."

def get_base64_image(uploaded_file):
    if uploaded_file and uploaded_file.name.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
        try:
            img = Image.open(uploaded_file)
            if img.mode != 'RGB': img = img.convert('RGB')
            img.thumbnail((1024, 1024))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except: return None
    return None

def extract_frames_from_video_base64(video_file, num_frames=6):
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
                h, w = frame.shape[:2]
                scale = min(1024/w, 1024/h)
                if scale < 1: frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                base64_frames.append(base64.b64encode(buffer).decode('utf-8'))
    cap.release()
    try: os.remove(tmp_path)
    except: pass
    return base64_frames

def auto_match_frame(engine, api_key, ep, frames_b64, room_name):
    """【核心黑科技】利用视觉大脑在多帧中自动寻找最匹配的房间"""
    if not frames_b64 or len(frames_b64) <= 1: return 0
    prompt_text = f"Identify which of these {len(frames_b64)} sequential images best represents the '{room_name}'. Reply ONLY with the single integer index (0 to {len(frames_b64)-1}). No other text."
    try:
        if engine == "Google Gemini 2.5":
            client = genai.Client(api_key=api_key)
            contents = [prompt_text]
            for b64 in frames_b64: contents.append(Image.open(io.BytesIO(base64.b64decode(b64))))
            res = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
            text = res.text
        else:
            client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=30.0)
            content_list = [{"type": "text", "text": prompt_text}]
            for b64 in frames_b64: content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            res = client.chat.completions.create(model=ep, messages=[{"role": "user", "content": content_list}], temperature=0.1)
            text = res.choices[0].message.content
        # 提取第一个数字
        nums = re.findall(r'\d+', text)
        if nums and 0 <= int(nums[0]) < len(frames_b64): return int(nums[0])
        return 0
    except: return 0

def append_media_to_doubao(content_list, uploaded_file, lang_code):
    if not uploaded_file: return
    if uploaded_file.name.lower().endswith(('mp4', 'mov', 'avi', 'gif')):
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

if "design_result" not in st.session_state: st.session_state.design_result = None
if "showhouse_b64s" not in st.session_state: st.session_state.showhouse_b64s = []
if "user_req_cache" not in st.session_state: st.session_state.user_req_cache = ""

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
doubao_draw_ep = st.sidebar.text_input(t["doubao_draw_ep"], placeholder="ep-... (Seedream)", help="用于渲染最终效果图")

st.title(t["title"])

col1, col2 = st.columns(2)
with col1:
    st.subheader(t["step1"])
    floor_plan_file = st.file_uploader("Floor Plan", type=["jpg", "png", "jpeg"], key="fp", label_visibility="collapsed")
    if floor_plan_file: st.image(floor_plan_file, width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)

with col2:
    st.subheader(t["step2"])
    st.caption(t["step2_desc"])
    showhouse_file = st.file_uploader("Showhouse Reality", type=["jpg", "png", "jpeg", "mp4", "gif", "mov"], key="sh", label_visibility="collapsed")
    if showhouse_file:
        if showhouse_file.name.lower().endswith(('mp4', 'mov', 'gif')): st.video(showhouse_file)
        else: st.image(showhouse_file, width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.subheader(t["step3"])
    st.caption(t["step3_desc"])
    style_file = st.file_uploader("Style", type=["jpg", "png", "jpeg"], key="sf", label_visibility="collapsed")
    if style_file: st.image(style_file, width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)

with col4:
    st.subheader(t["step4"])
    user_req = st.text_area("Requirements", placeholder=t["req_placeholder"], height=150, label_visibility="collapsed")

# ==========================================
# 4. 大脑处理逻辑 (全屋文本生成)
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
                st.session_state.user_req_cache = user_req 
                
                # 缓存实勘图用于后续生图
                if showhouse_file.name.lower().endswith(('mp4', 'mov', 'gif')):
                    st.session_state.showhouse_b64s = extract_frames_from_video_base64(showhouse_file, 6)
                else:
                    st.session_state.showhouse_b64s = [get_base64_image(showhouse_file)]
                
                if engine_choice == "Google Gemini 2.5":
                    client = genai.Client(api_key=api_key)
                    contents = [prompt_text]
                    contents.append(Image.open(floor_plan_file))
                    for b64 in st.session_state.showhouse_b64s: contents.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                    if has_style: contents.append(Image.open(style_file))
                    
                    res = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
                    st.session_state.design_result = res.text

                elif engine_choice == "字节豆包 (Doubao)":
                    client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=60.0)
                    content_list = [{"type": "text", "text": prompt_text}]
                    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(floor_plan_file)}"}})
                    for b64 in st.session_state.showhouse_b64s: content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    if has_style: content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(style_file)}"}})
                    
                    res = client.chat.completions.create(model=doubao_ep, messages=[{"role": "user", "content": content_list}])
                    st.session_state.design_result = res.choices[0].message.content

                st.success(t["success"])
            except Exception as e:
                st.error(f"大脑分析错误: {e}")

# ==========================================
# 5. 画笔处理逻辑 (AI 自动匹配 / 人工覆盖)
# ==========================================
if st.session_state.design_result:
    st.markdown("---")
    st.markdown(st.session_state.design_result)
    
    st.markdown("---")
    st.subheader("🖼️ AI 空间效果图渲染")
    
    col5, col6 = st.columns([1, 1])
    with col5:
        selected_room = st.selectbox(t["label_drawing_room"], t["option_drawing_room"])
        st.info(f"👁️ **AI 智能寻帧已就绪**。AI将自动从您上传的视频/图组中寻找【{selected_room}】。如果您想强制指定视角，也可以在下方手动上传底图。")
        room_reality_file = st.file_uploader(f"上传【{selected_room}】实景图 (选填/覆盖用)", type=["jpg", "png", "jpeg"], key="room_reality")
        draw_key = api_key if engine_choice == "字节豆包 (Doubao)" else st.sidebar.text_input("输入豆包 API Key 进行绘图", type="password")
        
    with col6:
        if room_reality_file:
            st.image(room_reality_file, caption=f"强制覆盖：以此作为【{selected_room}】骨架", width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)
            
    if st.button(t["btn_draw"], type="primary", use_container_width=True):
        if not draw_key or not doubao_draw_ep:
            st.error("⚠️ 请在左侧栏配置完整的 【豆包 API Key】 和 【豆包 画笔接入点 (Seedream EP)】")
        else:
            with st.spinner(t["status_drawing"]):
                try:
                    # 确定底图逻辑
                    current_reality_b64 = None
                    if room_reality_file:
                        st.toast("使用人工覆盖底图...", icon="✅")
                        current_reality_b64 = get_base64_image(room_reality_file)
                    else:
                        st.toast(f"正在启动视觉大脑，在视频帧中搜索【{selected_room}】...", icon="🧠")
                        matched_idx = auto_match_frame(engine_choice, api_key, doubao_ep, st.session_state.showhouse_b64s, selected_room)
                        st.toast(f"✅ AI 锁定第 {matched_idx + 1} 个画面作为结构骨架！", icon="🎯")
                        if st.session_state.showhouse_b64s: current_reality_b64 = st.session_state.showhouse_b64s[matched_idx]

                    if not current_reality_b64:
                        st.error("未找到有效底图！")
                        st.stop()
                        
                    draw_client = OpenAI(api_key=draw_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=60.0)
                    current_has_style = style_file is not None
                    
                    drawing_prompt_desc = get_drawing_prompt_desc(
                        selected_room, st.session_state.design_result, st.session_state.user_req_cache, current_has_style 
                    )
                    
                    ref_images = [f"data:image/jpeg;base64,{current_reality_b64}"]
                    style_b64 = get_base64_image(style_file)
                    if style_b64: ref_images.append(f"data:image/jpeg;base64,{style_b64}")
                    
                    imagesResponse = draw_client.images.generate(
                        model=doubao_draw_ep, 
                        prompt=drawing_prompt_desc, 
                        size="2K",
                        response_format="b64_json",
                        stream=True,
                        extra_body={
                            "image": ref_images,
                            "watermark": False,
                            "sequential_image_generation": "auto",
                            "sequential_image_generation_options": {"max_images": 1} 
                        }
                    )
                    
                    image_placeholder = st.empty()
                    for event in imagesResponse:
                        if event is None: continue
                        if event.type in ["image_generation.partial_succeeded", "image_generation.succeeded"]:
                            if event.b64_json:
                                image_data = base64.b64decode(event.b64_json)
                                image_placeholder.image(image_data, caption=f"✨ {selected_room} AI 渲染完成", width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)
                                
                    st.toast(t["success_draw"])
                    
                except Exception as e:
                    st.error(f"效果图渲染失败: {e}")
