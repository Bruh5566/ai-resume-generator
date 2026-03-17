import os
import json
import io
import re
import requests
import streamlit as st
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document


load_dotenv()

st.set_page_config(
    page_title="AI 求職投遞助手",
    page_icon="📄",
    layout="wide",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
print("DEBUG OPENAI:", OPENAI_API_KEY[:8] if OPENAI_API_KEY else "EMPTY")
print("DEBUG GROQ:", GROQ_API_KEY[:8] if GROQ_API_KEY else "EMPTY")

MODEL_OPTIONS = {
    "平衡版（推薦）- gpt-4.1-mini": "gpt-4.1-mini",
    "超省成本 - gpt-4.1-nano": "gpt-4.1-nano",
}

SYSTEM_PROMPT = """
你是一位專業的求職顧問與履歷優化專家。
你的任務是根據使用者提供的履歷內容、目標職缺與語氣需求，輸出高品質、具體、可直接使用的求職材料。

請遵守以下規則：
1. 全部使用繁體中文。
2. 不可捏造履歷中不存在的經歷、證照或技能。
3. 若履歷與職缺有落差，請用可轉移能力與學習意願做合理補強，但不能亂編。
4. 內容要具體、自然、專業，不要空泛。
5. 請只輸出 JSON，不要加任何前後說明。

輸出 JSON schema：
{
  "cover_letter": "客製化求職信全文",
  "self_intro_30s": "30秒自我介紹",
  "resume_optimization": [
    "建議1",
    "建議2",
    "建議3"
  ],
  "interview_questions": [
    {
      "question": "問題1",
      "answer_hint": "簡短作答方向1"
    }
  ]
}
"""


def read_pdf_bytes(file_bytes: bytes) -> str:
    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts).strip()
    except Exception as e:
        return f"[PDF 讀取失敗] {e}"


def read_docx_bytes(file_bytes: bytes) -> str:
    try:
        docx_stream = io.BytesIO(file_bytes)
        doc = Document(docx_stream)
        texts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(texts).strip()
    except Exception as e:
        return f"[DOCX 讀取失敗] {e}"


def read_txt_bytes(file_bytes: bytes) -> str:
    encodings_to_try = ["utf-8", "utf-8-sig", "cp950", "big5", "latin-1"]
    for enc in encodings_to_try:
        try:
            return file_bytes.decode(enc).strip()
        except Exception:
            continue
    return "[TXT 讀取失敗] 無法判斷編碼"


def parse_resume_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".pdf"):
        return read_pdf_bytes(file_bytes)
    elif file_name.endswith(".docx"):
        return read_docx_bytes(file_bytes)
    elif file_name.endswith(".txt"):
        return read_txt_bytes(file_bytes)
    else:
        return "[不支援的檔案格式，請上傳 PDF / DOCX / TXT]"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)

def normalize_job_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u3000", " ")
    text = text.replace("\r", "\n")

    # 移除過多空白與空行
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines).strip()


def clean_job_description(text: str) -> str:
    text = normalize_job_text(text)

    if not text:
        return ""

    noise_keywords = [
        "隱私權政策",
        "privacy policy",
        "terms of use",
        "cookie",
        "cookies",
        "copyright",
        "版權所有",
        "all rights reserved",
        "sign in",
        "log in",
        "登入",
        "註冊",
        "apply now",
        "立即應徵",
        "share",
        "分享",
        "檢舉",
        "advertisement",
        "廣告",
        "加入會員",
        "下載app",
        "download app",
        "追蹤我們",
        "follow us",
        "linkedin",
        "104人力銀行",
        "1111人力銀行",
        "yes123",
        "更多工作機會",
        "查看更多",
        "看過此職缺的人也看了",
        "推薦職缺",
        "公司介紹",
        "關於我們",
        "about us",
        "企業簡介",
        "聯絡我們",
        "contact us",
        "服務條款",
    ]

    useful_keywords = [
        "職缺",
        "職稱",
        "工作內容",
        "工作職責",
        "responsibilities",
        "requirements",
        "qualifications",
        "技能",
        "skills",
        "條件要求",
        "工作經驗",
        "學歷要求",
        "科系要求",
        "語文條件",
        "擅長工具",
        "其他條件",
        "待遇",
        "薪資",
        "salary",
        "location",
        "地點",
        "上班地點",
        "工作地點",
        "福利",
        "benefits",
        "職務類別",
        "工作性質",
    ]

    cleaned_lines = []
    seen = set()

    for line in text.splitlines():
        line_lower = line.lower().strip()

        if len(line_lower) < 2:
            continue

        # 太像雜訊的先丟掉
        if any(k.lower() in line_lower for k in noise_keywords):
            # 但如果同時含有很明顯的職缺資訊關鍵字，保留
            if not any(k.lower() in line_lower for k in useful_keywords):
                continue

        # 避免重複行
        if line_lower in seen:
            continue
        seen.add(line_lower)

        cleaned_lines.append(line)

    # 如果清洗太狠，把原文縮短版保底留著
    if not cleaned_lines:
        cleaned_lines = [line for line in text.splitlines() if len(line.strip()) > 4]

    cleaned_text = "\n".join(cleaned_lines)

    # 長度控制，避免垃圾文字吃掉 token
    return cleaned_text[:5000].strip()

def clean_518_job_text(text: str) -> str:
    if not text:
        return ""

    noise_starts = [
        "立即下載求職 App",
        "立即下載企業 App",
        "AI 幫你寫好自傳啦！",
        "會員專屬功能使用更便利！",
        "全新功能「技能交換」",
        "還想看看其他的嗎？",
        "前往APP還有更多技能等你探索～",
        "看過此職缺的人也看了...",
        "來「518熊班」APP",
        "518熊班 客服專線",
        "© 2026 by Addcn Technology Co., Ltd. All Rights Reserved.",
        "※518熊班提醒您",
    ]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []

    for line in lines:
        # 一旦進入明顯的雜訊區塊，後面通常整段都不用了
        if any(marker in line for marker in noise_starts):
            break
        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # 第二層過濾，移除殘留雜訊行
    remove_contains = [
        "求職 App",
        "企業 App",
        "技能交換",
        "看過此職缺的人也看了",
        "518熊班提醒您",
        "客服專線",
        "All Rights Reserved",
        "上櫃公司股票代碼",
        "許可證字號",
    ]

    final_lines = []
    for line in result.splitlines():
        if any(keyword in line for keyword in remove_contains):
            continue
        final_lines.append(line)

    return "\n".join(final_lines).strip()

def compress_job_description(text: str) -> str:
    """
    第二層壓縮：優先保留比較像 JD 的行。
    如果抓到的是一大坨混雜文字，這層很有用。
    """
    text = clean_job_description(text)

    if not text:
        return ""

    priority_keywords = [
        "職缺", "職稱", "公司", "工作內容", "工作職責", "responsibilities",
        "requirements", "qualifications", "技能", "skills", "條件要求",
        "工作經驗", "學歷要求", "科系要求", "語文條件", "擅長工具",
        "其他條件", "待遇", "薪資", "salary", "location", "地點",
        "上班地點", "工作地點", "福利", "benefits", "職務類別", "工作性質"
    ]

    lines = text.splitlines()
    priority_lines = []
    other_lines = []

    for line in lines:
        line_lower = line.lower()
        if any(k.lower() in line_lower for k in priority_keywords):
            priority_lines.append(line)
        else:
            other_lines.append(line)

    # 先保留最像職缺的內容，再補一些其他文字
    merged = priority_lines + other_lines[:30]
    merged_text = "\n".join(merged)

    return merged_text[:3500].strip()

def build_user_prompt(resume_text: str, job_desc: str, tone: str, extra_notes: str) -> str:
    return f"""
請根據以下資料，產出符合格式的 JSON。

【履歷內容】
{resume_text}

【職缺內容】
{job_desc}

【語氣風格】
{tone}

【補充需求】
{extra_notes if extra_notes.strip() else "無"}

請輸出：
1. cover_letter：一封可直接投遞的客製化求職信
2. self_intro_30s：30秒自我介紹
3. resume_optimization：至少3點履歷優化建議
4. interview_questions：請提供10題可能面試題，每題都附上簡短作答方向
""".strip()


def generate_ai(system_prompt: str, user_prompt: str, provider: str, model: str) -> dict:
    text = ""

    if provider == "OpenAI":
        if not OPENAI_API_KEY:
            raise ValueError("缺少 OPENAI_API_KEY。")

        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            text={"format": {"type": "json_object"}},
            temperature=0.4,
            max_output_tokens=2200,
        )

        text = response.output_text.strip()

    elif provider == "Groq":
        if not GROQ_API_KEY:
            raise ValueError("缺少 GROQ_API_KEY。")

        client = Groq(api_key=GROQ_API_KEY)

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )

        text = completion.choices[0].message.content.strip()

        text = text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        return json.loads(text)


def render_result(data: dict):
    st.subheader("📩 客製化求職信")
    st.write(data.get("cover_letter", ""))

    st.subheader("🎤 30秒自我介紹")
    st.write(data.get("self_intro_30s", ""))

    st.subheader("🛠️ 履歷優化建議")
    for item in data.get("resume_optimization", []):
        st.markdown(f"- {item}")

    st.subheader("❓ 可能面試題")
    questions = data.get("interview_questions", [])
    for idx, item in enumerate(questions, 1):
        question = item.get("question", "")
        hint = item.get("answer_hint", "")
        st.markdown(f"**{idx}. {question}**")
        st.write(f"作答方向：{hint}")

def fetch_url_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text
    except requests.exceptions.SSLError:
        # 先作為測試 fallback，用於某些憑證驗證怪異的站
        resp = requests.get(url, headers=headers, timeout=15, verify=False)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_jsonld_jobposting(soup: BeautifulSoup) -> str:
    scripts = soup.find_all("script", type="application/ld+json")
    chunks = []

    for script in scripts:
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            if item.get("@type") == "JobPosting":
                title = item.get("title", "")
                desc = item.get("description", "")
                skills = item.get("skills", "")
                qualifications = item.get("qualifications", "")
                responsibilities = item.get("responsibilities", "")
                hiring_org = ""

                org = item.get("hiringOrganization")
                if isinstance(org, dict):
                    hiring_org = org.get("name", "")

                parts = [
                    f"職稱：{title}" if title else "",
                    f"公司：{hiring_org}" if hiring_org else "",
                    f"工作內容：{BeautifulSoup(desc, 'lxml').get_text(' ', strip=True)}" if desc else "",
                    f"技能需求：{skills}" if skills else "",
                    f"資格條件：{qualifications}" if qualifications else "",
                    f"工作職責：{responsibilities}" if responsibilities else "",
                ]
                job_text = "\n".join([p for p in parts if p.strip()])
                if job_text.strip():
                    chunks.append(job_text)

    return "\n\n".join(chunks).strip()


def extract_104(soup: BeautifulSoup) -> str:
    title = ""
    content_parts = []

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        content_parts.append(desc_tag["content"])

    jsonld_text = extract_jsonld_jobposting(soup)
    if jsonld_text:
        content_parts.append(jsonld_text)

    result = clean_text("\n\n".join([p for p in content_parts if p.strip()]))

    # 如果前面抓到的內容太少，退回頁面純文字
    if len(result) < 200:
        body_text = soup.get_text("\n", strip=True)
        lines = [line.strip() for line in body_text.splitlines() if len(line.strip()) > 8]

        # 嘗試只保留比較像職缺資訊的行
        keywords = ["工作內容", "職務類別", "工作待遇", "上班地點", "工作經歷", "學歷要求", "科系要求", "擅長工具", "其他條件"]
        filtered = [line for line in lines if any(k in line for k in keywords)]

        fallback_text = "\n".join(filtered[:120]) if filtered else "\n".join(lines[:120])

        result = clean_text(
            "\n".join(
                [p for p in [f"職缺標題：{title}" if title else "", fallback_text[:4000]] if p.strip()]
            )
        )

    return result


def extract_1111(soup: BeautifulSoup) -> str:
    title = ""
    content_parts = []

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        content_parts.append(desc_tag["content"])

    jsonld_text = extract_jsonld_jobposting(soup)
    if jsonld_text:
        content_parts.append(jsonld_text)

    return clean_text(
        "\n".join(
            [p for p in [f"職缺標題：{title}" if title else "", *content_parts] if p.strip()]
        )
    )


def extract_yes123(soup: BeautifulSoup) -> str:
    title = ""
    content_parts = []

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        content_parts.append(desc_tag["content"])

    jsonld_text = extract_jsonld_jobposting(soup)
    if jsonld_text:
        content_parts.append(jsonld_text)

    return clean_text(
        "\n".join(
            [p for p in [f"職缺標題：{title}" if title else "", *content_parts] if p.strip()]
        )
    )

def extract_518(soup: BeautifulSoup) -> str:
    title = ""
    content_parts = []

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        content_parts.append(desc_tag["content"])

    jsonld_text = extract_jsonld_jobposting(soup)
    if jsonld_text:
        content_parts.append(jsonld_text)

    body_text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in body_text.splitlines() if len(line.strip()) > 1]
    body_joined = "\n".join(lines[:250])  # 先抓前段，避免整頁太長

    merged = "\n\n".join(
        [p for p in [f"職缺標題：{title}" if title else "", *content_parts, body_joined] if p.strip()]
    )

    merged = clean_text(merged)
    merged = clean_518_job_text(merged)

    return merged[:4000].strip()

def extract_linkedin(soup: BeautifulSoup) -> str:
    title = ""
    company = ""
    content_parts = []

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]

    desc_tag = soup.find("meta", property="og:description")
    if desc_tag and desc_tag.get("content"):
        content_parts.append(desc_tag["content"])

    jsonld_text = extract_jsonld_jobposting(soup)
    if jsonld_text:
        content_parts.append(jsonld_text)

    return clean_text(
        "\n".join(
            [p for p in [f"職缺標題：{title}" if title else "", f"公司名稱：{company}" if company else "", *content_parts] if p.strip()]
        )
    )


def extract_generic_job(soup: BeautifulSoup) -> str:
    title = ""
    description = ""

    title_tag = soup.find("meta", property="og:title")
    if title_tag and title_tag.get("content"):
        title = title_tag["content"]
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        description = desc_tag["content"]

    jsonld_text = extract_jsonld_jobposting(soup)
    if jsonld_text:
        return clean_text(
            "\n".join(
                [p for p in [f"職缺標題：{title}" if title else "", description, jsonld_text] if p.strip()]
            )
        )

    body_text = soup.get_text("\n", strip=True)
    body_text = "\n".join(line for line in body_text.splitlines() if len(line.strip()) > 8)

    parts = [
        f"職缺標題：{title}" if title else "",
        f"頁面描述：{description}" if description else "",
        body_text[:4000],  # 避免抓太長把 token 撐爆
    ]
    return clean_text("\n".join([p for p in parts if p.strip()]))


def extract_job_from_url(url: str) -> str:
    html = fetch_url_html(url)
    soup = BeautifulSoup(html, "lxml")
    url_lower = url.lower()

    page_text = soup.get_text(" ", strip=True)

    if "vue-start doesn't work properly without javascript enabled" in page_text.lower():
        raise ValueError("此網站內容需 JavaScript 載入，暫時無法直接抓取，請手動貼上職缺內容。")

    if "1111.com.tw" in url_lower:
        result = extract_1111(soup)
    elif "yes123.com.tw" in url_lower:
        result = extract_yes123(soup)
    elif "518.com.tw" in url_lower:
        result = extract_generic_job(soup) 
    elif "104.com.tw" in url_lower:
        raise ValueError("104 目前不支援自動抓取。")   
    elif "linkedin.com" in url_lower:
        raise ValueError("LinkedIn 目前不支援自動抓取。")
    else:
        result = extract_generic_job(soup)

    print("DEBUG URL:", url)
    print("DEBUG RESULT LEN:", len(result))
    print("DEBUG RESULT PREVIEW:", result[:300] if result else "EMPTY") 

    return clean_text(result)
        
    
def detect_noisy_job_source(job_text: str, url: str):
    """
    檢查職缺內容是否可能包含過多雜訊或非職缺資訊。
    回傳 (is_noisy: bool, message: str)
    """
    noisy_patterns = [
        "公司介紹", "關於我們", "about us", "企業簡介", "聯絡我們", "contact us", "服務條款",
        "隱私權政策", "privacy policy", "terms of use", "cookie", "cookies", "copyright",
        "版權所有", "all rights reserved", "sign in", "log in", "登入", "註冊", "apply now",
        "立即應徵", "share", "分享", "檢舉", "advertisement", "廣告", "加入會員", "下載app",
        "download app", "追蹤我們", "follow us", "linkedin", "更多工作機會", "查看更多",
        "看過此職缺的人也看了", "推薦職缺"
    ]
    noisy_count = sum(1 for k in noisy_patterns if k.lower() in job_text.lower())
    if noisy_count >= 5:
        return True, "偵測到職缺內容可能包含過多雜訊（如公司介紹、廣告、頁尾等），建議手動檢查並刪除無關內容。"
    return False, ""
    


def build_export_text(data: dict) -> str:
    lines = []
    lines.append("【客製化求職信】")
    lines.append(data.get("cover_letter", ""))
    lines.append("")
    lines.append("【30秒自我介紹】")
    lines.append(data.get("self_intro_30s", ""))
    lines.append("")
    lines.append("【履歷優化建議】")
    for item in data.get("resume_optimization", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("【可能面試題】")
    for idx, item in enumerate(data.get("interview_questions", []), 1):
        q = item.get("question", "")
        a = item.get("answer_hint", "")
        lines.append(f"{idx}. {q}")
        lines.append(f"   作答方向：{a}")
    return "\n".join(lines)


st.title("📄 AI 求職投遞助手")
st.caption("上傳履歷、貼上職缺，自動產生求職信、自介、履歷優化建議與面試題。")

if not OPENAI_API_KEY and not GROQ_API_KEY:
    st.error("找不到任何 API KEY，請在 .env 設定 OPENAI_API_KEY 或 GROQ_API_KEY")
    st.stop()

with st.sidebar:
    st.header("⚙️ 設定")

    provider = st.selectbox(
        "AI 提供商",
        ["Groq", "OpenAI"],
        index=0,
    )

    if provider == "Groq":
        model = st.selectbox(
            "模型",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant"
            ],
            index=0,
        )
    else:
        model = st.selectbox(
            "模型",
            [
                "gpt-4.1-mini",
                "gpt-4.1-nano"
            ],
            index=0,
        )

    tone = st.selectbox(
        "語氣風格",
        ["正式專業", "自然誠懇", "積極有企圖心", "簡潔俐落"],
        index=0,
    )

    extra_notes = st.text_area(
        "補充需求（可空白）",
        placeholder="例如：希望強調實習經驗、偏好科技業、避免太浮誇",
        height=120,
    )
    
col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader(
        "上傳履歷（PDF / DOCX / TXT）",
        type=["pdf", "docx", "txt"],
    )

    resume_text_manual = st.text_area(
        "或直接貼上履歷文字",
        placeholder="如果你懶得上傳檔案，也可以直接貼這裡",
        height=280,
    )

with col2:
    job_url = st.text_input(
        "貼上職缺網址（支援 1111 / yes123 / 部分一般公司職缺頁）",
        placeholder="例如：https://www.1111.com.tw/job/xxxxxx",
    )

    st.caption("系統會自動清洗抓到的職缺內容，盡量移除公司介紹、廣告、頁尾與其他雜訊。")
    st.caption("部分網站（例如 104、LinkedIn）暫不支援自動抓取，仍需要手動貼上職缺內容。")

    col2a, col2b = st.columns([1, 2])

    with col2a:
        fetch_job_btn = st.button("🌐 抓取職缺內容", use_container_width=True)

    with col2b:
        clear_job_btn = st.button("🧹 清空職缺內容", use_container_width=True)

    if "job_desc" not in st.session_state:
        st.session_state.job_desc = ""

    if fetch_job_btn:
        if not job_url.strip():
            st.warning("請先輸入職缺網址。")
        else:
            with st.spinner("正在抓取職缺內容..."):
                try:
                    extracted = extract_job_from_url(job_url.strip())
                    if extracted.strip():
                        st.session_state.job_desc = extracted
                        st.write(f"抓取長度：{len(extracted)}")
                        st.code(extracted[:500] if extracted else "EMPTY")
                        st.success("已成功抓取職缺內容。")
                        is_noisy, noisy_msg = detect_noisy_job_source(extracted, job_url.strip())
                        if is_noisy:
                            st.warning(noisy_msg)
                    else:
                        st.session_state.job_desc = ""
                        st.warning("抓到了頁面，但沒有成功解析出有效職缺內容，請手動貼上。")
                except Exception as e:
                    msg = str(e).lower()
                    st.session_state.job_desc = ""

                    if "104.com.tw" in job_url.lower():
                        st.warning("104 目前不支援自動抓取，請手動貼上職缺內容。")
                    elif "linkedin.com" in job_url.lower():
                        st.warning("LinkedIn 目前不支援自動抓取，請手動貼上職缺內容。")
                    elif "javascript" in msg:
                        st.warning("此網站目前不支援自動抓取，請手動貼上職缺內容。")
                    else:
                        st.error(f"抓取失敗：{repr(e)}")

    if clear_job_btn:
        st.session_state.job_desc = ""

    job_desc = st.text_area(
        "貼上職缺內容（JD）",
        key="job_desc",
        placeholder="你可以手動貼上，或先貼網址讓系統自動抓取",
        height=430,
    )
    
    if st.checkbox("顯示清洗後的 JD 預覽", value=False):
        cleaned_preview = compress_job_description(st.session_state.job_desc if "job_desc" in st.session_state else job_desc)
        st.text_area(
            "清洗後 JD 預覽",
            value=cleaned_preview,
            height=250,
            disabled=True,
        )

resume_text_file = parse_resume_file(uploaded_resume) if uploaded_resume else ""
resume_text = resume_text_manual.strip() if resume_text_manual.strip() else resume_text_file.strip()

with st.expander("📊 粗略成本預估（非精準計費）", expanded=False):
    input_text_preview = f"{resume_text}\n{job_desc}\n{extra_notes}\n{tone}"
    est_input_tokens = estimate_tokens(input_text_preview)
    st.write(f"估計輸入 tokens：約 {est_input_tokens:,}")

if st.button("🚀 生成求職材料", use_container_width=True):
    if not resume_text:
        st.warning("請先上傳履歷或貼上履歷內容。")
        st.stop()

    if not job_desc.strip():
        st.warning("請先貼上職缺內容。")
        st.stop()

    cleaned_job_desc = compress_job_description(job_desc.strip())

    user_prompt = build_user_prompt(
        resume_text=resume_text,
        job_desc=job_desc.strip(),
        tone=tone,
        extra_notes=extra_notes.strip(),
    )

    with st.spinner("AI 正在生成內容中..."):
        try:
            result = generate_ai(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                provider=provider,
                model=model,
                    )

            st.success("生成完成。")
            render_result(result)

            export_text = build_export_text(result)
            st.download_button(
                label="⬇️ 下載 TXT 結果",
                data=export_text.encode("utf-8"),
                file_name="job_materials.txt",
                mime="text/plain",
            )

            with st.expander("查看原始 JSON"):
                st.json(result)

        except Exception as e:
            st.error(f"生成失敗：{repr(e)}")