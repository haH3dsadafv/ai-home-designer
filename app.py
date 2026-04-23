import streamlit as st
from google import genai
import tempfile
from PIL import Image

# ==========================================
# 1. 语言字典 (i18n)
# ==========================================
LANG = {
    "CN": {
        "title": "🏠 智能户型设计助手 Pro",
        "sidebar_title": "⚙️ 系统设置",
        "api_key_label": "请输入 Gemini API Key",
        "step1": "📍 步骤 1: 上传户型图 (必填项)",
        "step2": "🏢 步骤 2: 样板间/现场实拍 (了解空间)",
        "step2_desc": "上传开发商样板间或毛坯房实拍，帮助AI理解真实采光与空间感 (视频/动图/图片)",
        "step3": "🎨 步骤 3: 期望风格参考 (选填项)",
        "step3_desc": "上传您真正喜欢的装修风格参考。如果不传，AI将根据您的文字描述自由发挥",
        "step4": "💬 步骤 4: 具体改造需求",
        "req_placeholder": "例如：想要现代极简风格，主卧需要衣帽间，样板间原有的猪肝红地板必须换掉...",
        "btn_generate": "🪄 开始智能设计",
        "warning_api": "请在侧边栏输入 API Key！",
        "warning_file": "请至少上传户型图和样板间参考！",
        "status_analyzing": "正在深度解析空间结构与视频素材，生成方案中...请稍候...",
        "res_layout": "📍 空间结构与实勘分析",
        "res_design": "🛋️ 定制化设计方案",
        "res_prompt": "🎨 AI 效果图提示词 (Midjourney/SD)",
        "success": "设计完成！"
    },
    "EN": {
        "title": "🏠 AI Home Designer Pro",
        "sidebar_title": "⚙️ Settings",
        "api_key_label": "Enter Gemini API Key",
        "step1": "📍 Step 1: Floor Plan (Required)",
        "step2": "🏢 Step 2: Showhouse / Reality (Understand Space)",
        "step2_desc": "Upload showhouse or actual room footage to help AI understand lighting and volume (Video/GIF/Image)",
        "step3": "🎨 Step 3: Desired Style (Optional)",
        "step3_desc": "Upload a reference for the style you actually want. If skipped, AI will rely on your text.",
        "step4": "💬 Step 4: Specific Requirements",
        "req_placeholder": "e.g., Modern minimalist, walk-in closet in master bed, remove the original dark wood floors...",
        "btn_generate": "🪄 Generate Design",
        "warning_api": "Please enter your API Key in the sidebar!",
        "warning_file": "Please upload at least a floor plan and showhouse reference!",
        "status_analyzing": "Analyzing spatial structure and video footage, generating... Please wait...",
        "res_layout": "📍 Spatial & Reality Analysis",
        "res_design": "🛋️ Custom Design Plan",
        "res_prompt": "🎨 AI Rendering Prompts (Midjourney/SD)",
        "success": "Design Completed!"
    }
}

# ==========================================
# 2. 媒体处理辅助函数 (适配新版 API)
# ==========================================
def process_uploaded_file(client, uploaded_file):
    """根据文件类型，转化为 Gemini 可接受的格式"""
    if uploaded_file is None:
        return None
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # 处理图片 (PIL Image 可以直接传给新版 SDK)
    if file_extension in ['jpg', 'jpeg', 'png', 'webp']:
        return Image.open(uploaded_file)
    
    # 处理视频/动图 (存入临时文件供 Client 上传)
    elif file_extension in ['mp4', 'mov', 'avi', 'gif']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # 使用新版 SDK 上传文件
        uploaded_video = client.files.upload(file=tmp_file_path)
        return uploaded_video
    
    return None

# ==========================================
# 3. 核心提示词工程 (解耦空间与风格)
# ==========================================
def get_system_prompt(language, user_requirements, has_style_ref):
    style_logic_cn = "我提供了【期望风格参考图/视频】，请严格按照该参考的色调、材质和软装氛围进行设计。" if has_style_ref else "我没有提供具体的风格参考图，请完全根据我的【具体改造需求】文字描述来构思风格。"
    style_logic_en = "I have provided a [Desired Style Reference]. Please strictly follow its color palette, materials, and vibe." if has_style_ref else "I have not provided a style reference image. Please conceptualize the style entirely based on my [Specific Requirements] text."

    if language == "CN":
        return f"""
        你是一名精通中国及全球房地产市场的顶尖全栈室内设计师和 AI 提示词专家。
        我现在为你提供多份视觉资料，请按顺序理解它们的作用：
        1. 【户型图】：用于理解房屋的硬性尺寸、墙体结构和房间分布。
        2. 【样板间/实勘参考】：这是房屋交付时的真实物理状态。请用它来理解真实的空间层高、自然采光、窗户位置和空间体积感，**但不要受其原有老旧或开发商固有装修风格的局限**。
        3. 【期望风格参考】：{style_logic_cn}
        
        客户的具体改造需求是：【{user_requirements}】
        
        请结合上述所有信息，输出三大板块内容（请使用清晰的 Markdown 格式，不要使用复杂的代码块或 LaTeX）：
        
        ### 📍 空间结构与实勘分析
        (综合户型图和样板间画面，分析空间的真实采光、动线优缺点。指出样板间中哪些硬装/格局是可以保留的，哪些是必须拆改的)
        
        ### 🛋️ 定制化设计方案
        (将客户想要的风格“注入”到真实的样板间空间中。分房间详细说明：墙地面材质变更、色彩搭配、灯光设计以及核心家具选型)
        
        ### 🎨 AI 效果图提示词
        (为主要房间生成用于 Midjourney 或 Stable Diffusion 的全英文提示词。必须包含具体的空间结构描述+客户想要的风格词汇。格式要求：[Room name] interior design, [specific style], [colors], [materials], [lighting], highly detailed, 8k, photorealistic --ar 16:9)
        """
    else:
        return f"""
        You are a top-tier interior designer and AI prompt expert.
        Understand the roles of the provided visual inputs:
        1. [Floor Plan]: For understanding dimensions, structure, and layout.
        2. [Showhouse/Reality Reference]: This shows the actual physical state. Use it to understand ceiling height, natural lighting, and spatial volume, **but ignore its existing interior decoration style**.
        3. [Style Reference]: {style_logic_en}
        
        The client's specific requirement is: [{user_requirements}]
        
        Output the following three sections in clear Markdown:
        
        ### 📍 Spatial & Reality Analysis
        (Analyze the actual space combining the floor plan and showhouse footage. Highlight pros/cons of lighting and flow, and suggest what to keep or demolish.)
        
        ### 🛋️ Custom Design Plan
        (Inject the desired style into the actual physical space. Provide specific room-by-room recommendations for materials, colors, and furniture.)
        
        ### 🎨 AI Rendering Prompts
        (Generate English prompts for Midjourney/SD. Combine the spatial reality with the target style. Format: [Room name] interior design, [style], [colors], [materials], [lighting], highly detailed, 8k, photorealistic --ar 16:9)
        """

# ==========================================
# 4. Streamlit 界面逻辑
# ==========================================
st.set_page_config(page_title="AI Home Designer Pro", layout="wide")

st.sidebar.selectbox("Language / 语言", ["CN", "EN"], key="lang")
lang_code = st.session_state.lang
t = LANG[lang_code]

st.sidebar.header(t["sidebar_title"])
api_key = st.sidebar.text_input(t["api_key_label"], type="password", help="从 Google AI Studio 获取")

st.title(t["title"])

col1, col2 = st.columns(2)
with col1:
    st.subheader(t["step1"])
    floor_plan_file = st.file_uploader("Upload Floor Plan", type=["jpg", "png", "jpeg"], key="fp", label_visibility="collapsed")
    if floor_plan_file:
        st.image(floor_plan_file, use_container_width=True)

with col2:
    st.subheader(t["step2"])
    st.caption(t["step2_desc"])
    showhouse_file = st.file_uploader("Upload Showhouse", type=["jpg", "png", "jpeg", "mp4", "gif", "mov"], key="sh", label_visibility="collapsed")
    if showhouse_file:
        if showhouse_file.name.lower().endswith(('mp4', 'mov')):
            st.video(showhouse_file)
        else:
            st.image(showhouse_file, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.subheader(t["step3"])
    st.caption(t["step3_desc"])
    style_file = st.file_uploader("Upload Style", type=["jpg", "png", "jpeg", "mp4", "gif"], key="sf", label_visibility="collapsed")
    if style_file:
        if style_file.name.lower().endswith(('mp4', 'mov')):
            st.video(style_file)
        else:
            st.image(style_file, use_container_width=True)

with col4:
    st.subheader(t["step4"])
    user_req = st.text_area("Requirements", placeholder=t["req_placeholder"], height=150, label_visibility="collapsed")

# ==========================================
# 5. 核心处理逻辑 (全新 Gemini 2.5 API)
# ==========================================
if st.button(t["btn_generate"], type="primary", use_container_width=True):
    if not api_key:
        st.error(t["warning_api"])
    elif not floor_plan_file or not showhouse_file:
        st.error(t["warning_file"])
    else:
        with st.spinner(t["status_analyzing"]):
            try:
                # 1. 初始化新版 Client
                client = genai.Client(api_key=api_key)
                
                # 2. 构建输入内容列表
                has_style = style_file is not None
                prompt_text = get_system_prompt(lang_code, user_req, has_style)
                contents = [prompt_text]
                
                # 3. 处理并添加媒体文件 (注意这里将 client 传入以便上传视频)
                fp_media = process_uploaded_file(client, floor_plan_file)
                if fp_media: contents.append(fp_media)
                
                sh_media = process_uploaded_file(client, showhouse_file)
                if sh_media: contents.append(sh_media)
                
                if has_style:
                    style_media = process_uploaded_file(client, style_file)
                    if style_media: contents.append(style_media)
                
                # 4. 调用最新模型 gemini-2.5-flash
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents
                )
                
                # 展示结果
                st.success(t["success"])
                st.markdown("---")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"发生错误 / Error: {e}")
