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
        "title": "🏠 智能户型设计助手 (纯净旗舰版)",
        "sidebar_title": "⚙️ 系统设置",
        "engine_label": "🤖 选择大脑引擎 (文本分析)",
        "gemini_key": "Gemini API Key (海外节点)",
        "doubao_key": "豆包 API Key (火山引擎通用)",
        "doubao_ep": "豆包 大脑接入点 (Vision Pro EP)",
        "doubao_draw_ep": "豆包 画笔接入点 (Seedream EP)",
        "step1": "📍 步骤 1: 上传户型图 (必填项)",
        "step2": "🏢 步骤 2: 样板间实勘图库 (选填项)",
        "step2_desc": "可传现场照片/视频供AI参考光线层高。若不传，AI将凭空自由设计。",
        "step3": "🎨 步骤 3: 期望风格参考 (选填项)",
        "step3_desc": "上传您真正喜欢的装修风格参考图。",
        "step4": "💬 步骤 4: 具体改造需求",
        "req_placeholder": "例如：现代中古，主卧需要衣帽间，家里有猫...",
        "btn_generate": "🪄 生成全屋分析与设计方案",
        "label_drawing_room": "选择要绘制效果图的房间:",
        "option_drawing_room": ["大横厅 (Living Room)", "主卧 (Master Bedroom)", "次卧 (Bedroom 2)", "厨房 (Kitchen)", "电梯厅 (Elevator Hall)"],
        "btn_draw": "🎨 智能渲染 3D 效果图",
        "warning_api": "请在侧边栏完善大脑和画笔的 API 密钥信息！",
        "warning_file": "请至少上传一张【户型图】！",
        "info_extracting": "🎥 正在处理实勘多媒体矩阵...",
        "status_analyzing": "AI 大脑正在解析空间结构与尺寸，生成全屋方案中...",
        "status_drawing": "AI 画笔正在绘制效果图 (约 15 秒)...",
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
        "step2": "🏢 Step 2: Showhouse Gallery (Optional)",
        "step2_desc": "Upload photos/videos for lighting ref. If omitted, AI designs freely.",
        "step3": "🎨 Step 3: Desired Style (Optional)",
        "step3_desc": "Upload a reference for the desired style.",
        "step4": "💬 Step 4: Specific Requirements",
        "req_placeholder": "e.g., Modern minimalist, walk-in closet...",
        "btn_generate": "🪄 Generate Global Analysis & Plan",
        "label_drawing_room": "Select Room to Render:",
        "option_drawing_room": ["Living Room", "Master Bedroom", "Bedroom 2", "Kitchen", "Elevator Hall"],
        "btn_draw": "🎨 Render 3D Image",
        "warning_api": "Please configure Brain and Drawing API keys in the sidebar!",
        "warning_file": "Please upload at least a Floor Plan!",
        "info_extracting": "🎥 Processing media gallery...",
        "status_analyzing": "Analyzing the space & dimensions... Please wait...",
        "status_drawing": "Rendering 3D image (approx. 15s)...",
        "success": "✨ Global Design Plan Completed!",
        "success_draw": "🖼️ Rendering Completed!"
    }
}

# ==========================================
# 2. 核心提示词与辅助函数 
# ==========================================
def get_system_prompt(language, user_requirements, has_style_ref, has_reality):
    style_logic = "严格参考【期望风格图】的色调和材质。" if has_style_ref else "自行构思风格。"
    reality_logic = "参考【实勘图库】中的真实层高和采光。" if has_reality else "客户未提供实勘图，请完全基于【户型图】的尺寸和常规逻辑，自由发挥空间想象力进行设计。"
    
    if language == "CN":
        return f"你是一名顶尖室内设计师。\n1. 【全局户型图】：看结构。**务必读取尺寸数值作为约束**。\n2. 【现场实勘】：{reality_logic}\n3. 【期望风格】：{style_logic}\n\n客户需求：【{user_requirements}】\n\n请输出：\n### 📍 空间尺寸与分割分析\n### 🛋️ 定制化全屋设计方案\n### 🎨 AI 效果图描述词 (英文 Midjourney格式)"
    else:
        return f"You are a top interior designer.\nAnalyze structure from Floor Plan. Reality: {reality_logic}\nStyle: {style_logic}\nNeeds: [{user_requirements}]\n\nOutput:\n### 📍 Spatial Analysis\n### 🛋️ Custom Global Design Plan\n### 🎨 AI Rendering Prompts"

def get_drawing_prompt_desc(selected_room, plan_context, user_requirements, has_style_ref, layout_constraint):
    return f"Highly detailed, photorealistic interior rendering of the {selected_room}. {layout_constraint} Plan context: {plan_context[:300]}... User needs: {user_requirements}. Trending on ArtStation, 8k resolution, Unreal Engine 5 render, cinematic lighting."

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

def extract_frames_from_video_base64(video_file, num_frames=4):
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

# ==========================================
# 3. Streamlit 界面及状态初始化
# ==========================================
st.set_page_config(page_title="AI Home Designer", layout="wide")

if "design_result" not in st.session_state: st.session_state.design_result = None
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
    # 【已修改】为豆包接入内置了默认的 API Key 和 大脑 EP
    api_key = st.sidebar.text_input(
        t["doubao_key"], 
        value="ark-f1e28220-55e9-4925-b43c-356effbaaf83-7c2fc", 
        type="password"
    )
    doubao_ep = st.sidebar.text_input(
        t["doubao_ep"], 
        value="ep-20260423170602-t7ps4", 
        placeholder="ep-... (Vision Pro)"
    )

st.sidebar.markdown("---")
# 【已修改】为绘图引擎内置了默认的 画笔 EP
doubao_draw_ep = st.sidebar.text_input(
    t["doubao_draw_ep"], 
    value="ep-20260423203519-7vc8l", 
    placeholder="ep-... (Seedream)", 
    help="用于渲染最终效果图"
)

st.title(t["title"])

col1, col2 = st.columns(2)
with col1:
    st.subheader(t["step1"])
    floor_plan_file = st.file_uploader("Floor Plan (必填)", type=["jpg", "png", "jpeg"], key="fp", label_visibility="collapsed")
    if floor_plan_file: st.image(floor_plan_file, width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)

with col2:
    st.subheader(t["step2"])
    st.caption(t["step2_desc"])
    showhouse_files = st.file_uploader("Showhouse Reality (选填)", type=["jpg", "png", "jpeg", "mp4", "gif", "mov"], accept_multiple_files=True, key="sh", label_visibility="collapsed")
    
    if showhouse_files:
        st.success(f"✅ 成功加载 {len(showhouse_files)} 个实景文件素材")
        first_file = showhouse_files[0]
        if first_file.name.lower().endswith(('mp4', 'mov', 'gif')): st.video(first_file)
        else: st.image(first_file, width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.subheader(t["step3"])
    st.caption(t["step3_desc"])
    style_file = st.file_uploader("Style (选填)", type=["jpg", "png", "jpeg"], key="sf", label_visibility="collapsed")
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
    elif not floor_plan_file: 
        st.error(t["warning_file"])
    else:
        with st.spinner(t["status_analyzing"]):
            try:
                has_style = style_file is not None
                has_reality = showhouse_files is not None and len(showhouse_files) > 0
                prompt_text = get_system_prompt(lang_code, user_req, has_style, has_reality)
                st.session_state.user_req_cache = user_req 
                
                showhouse_b64s = []
                if has_reality:
                    st.toast(t["info_extracting"], icon="⏳")
                    for f in showhouse_files:
                        if f.name.lower().endswith(('mp4', 'mov', 'gif')):
                            showhouse_b64s.extend(extract_frames_from_video_base64(f, 4))
                        else:
                            b64 = get_base64_image(f)
                            if b64: showhouse_b64s.append(b64)
                    showhouse_b64s = showhouse_b64s[:10] # 限流保护
                
                if engine_choice == "Google Gemini 2.5":
                    client = genai.Client(api_key=api_key)
                    contents = [prompt_text]
                    contents.append(Image.open(floor_plan_file))
                    for b64 in showhouse_b64s: contents.append(Image.open(io.BytesIO(base64.b64decode(b64))))
                    if has_style: contents.append(Image.open(style_file))
                    
                    res = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
                    st.session_state.design_result = res.text

                elif engine_choice == "字节豆包 (Doubao)":
                    client = OpenAI(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=60.0)
                    content_list = [{"type": "text", "text": prompt_text}]
                    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(floor_plan_file)}"}})
                    
                    for b64 in showhouse_b64s: 
                        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        
                    if has_style: content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_base64_image(style_file)}"}})
                    
                    st.toast("大脑正在解析户型尺寸写方案...", icon="🧠")
                    res = client.chat.completions.create(model=doubao_ep, messages=[{"role": "user", "content": content_list}])
                    st.session_state.design_result = res.choices[0].message.content

                st.success(t["success"])
            except Exception as e:
                st.error(f"大脑分析错误: {e}")

# ==========================================
# 5. 画笔处理逻辑 (纯净结构绑定 / 凭空想象)
# ==========================================
if st.session_state.design_result:
    st.markdown("---")
    st.markdown(st.session_state.design_result)
    
    st.markdown("---")
    st.subheader("🖼️ AI 空间效果图渲染")
    
    col5, col6 = st.columns([1, 1])
    with col5:
        selected_room = st.selectbox(t["label_drawing_room"], t["option_drawing_room"])
        st.info(f"💡 **自由模式**：如果您想完全约束门窗结构，请上传【{selected_room}】的照片。如果不传，AI将基于户型图尺寸自由发挥。")
        room_reality_file = st.file_uploader(f"上传【{selected_room}】实景底图 (选填)", type=["jpg", "png", "jpeg"], key="room_reality")
        
        # 【已修改】如果大脑选择 Gemini，画笔仍需使用内置的豆包默认 Key
        draw_key = api_key if engine_choice == "字节豆包 (Doubao)" else st.sidebar.text_input(
            "输入豆包 API Key 进行绘图", 
            value="ark-f1e28220-55e9-4925-b43c-356effbaaf83-7c2fc", 
            type="password"
        )
        
    with col6:
        if room_reality_file:
            st.image(room_reality_file, caption=f"结构约束：以此图作为【{selected_room}】骨架", width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)
            
    if st.button(t["btn_draw"], type="primary", use_container_width=True):
        if not draw_key or not doubao_draw_ep:
            st.error("⚠️ 请在左侧栏配置完整的 【豆包 API Key】 和 【豆包 画笔接入点 (Seedream EP)】")
        else:
            with st.spinner(t["status_drawing"]):
                try:
                    draw_client = OpenAI(api_key=draw_key, base_url="https://ark.cn-beijing.volces.com/api/v3", timeout=60.0)
                    
                    # 确定底图逻辑与布局约束提示词
                    ref_images = []
                    if room_reality_file:
                        st.toast(f"提取【{selected_room}】物理结构中...", icon="📐")
                        current_reality_b64 = get_base64_image(room_reality_file)
                        ref_images.append(f"data:image/jpeg;base64,{current_reality_b64}")
                        layout_constraint = "Strictly preserve the exact structural layout, walls, windows, and doors from the uploaded reality image."
                    else:
                        st.toast(f"进入自由构思模式，根据户型图想象【{selected_room}】...", icon="✨")
                        layout_constraint = "No reality image provided. Freely and logically design the spatial layout of this room based entirely on the dimensions and overall floor plan concept."

                    current_has_style = style_file is not None
                    drawing_prompt_desc = get_drawing_prompt_desc(
                        selected_room, st.session_state.design_result, st.session_state.user_req_cache, current_has_style, layout_constraint
                    )
                    
                    style_b64 = get_base64_image(style_file)
                    if style_b64: ref_images.append(f"data:image/jpeg;base64,{style_b64}")
                    
                    # 组装API参数，若没有任何图，则传 None (纯文生图)
                    extra_body = {
                        "watermark": False,
                        "sequential_image_generation": "auto",
                        "sequential_image_generation_options": {"max_images": 1}
                    }
                    if ref_images:
                        extra_body["image"] = ref_images
                        
                    imagesResponse = draw_client.images.generate(
                        model=doubao_draw_ep, 
                        prompt=drawing_prompt_desc, 
                        size="2K",
                        response_format="b64_json",
                        stream=True,
                        extra_body=extra_body
                    )
                    
                    image_placeholder = st.empty()
                    for event in imagesResponse:
                        if event is None: continue
                        if event.type in ["image_generation.partial_succeeded", "image_generation.succeeded"]:
                            if event.b64_json:
                                image_data = base64.b64decode(event.b64_json)
                                image_placeholder.image(image_data, caption=f"✨ {selected_room} 渲染完成", width="stretch" if st.__version__ >= "1.30.0" else None, use_column_width=True)
                                
                    st.toast(t["success_draw"])
                    
                except Exception as e:
                    st.error(f"效果图渲染失败: {e}")
