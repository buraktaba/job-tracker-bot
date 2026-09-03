import os
import time
import json
import smtplib
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from typing import List
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from google.genai import types
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# ==========================================
# 1. AYARLAR VE BAĞLANTILAR
# ==========================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", SENDER_EMAIL)
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "65"))
DB_FILE = "processed_jobs.json"

if not GEMINI_API_KEY or not SENDER_EMAIL or not EMAIL_PASSWORD:
    raise ValueError("Gerekli çevre değişkenleri (GEMINI_API_KEY, SENDER_EMAIL, EMAIL_PASSWORD) eksik!")

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================================
# 2. HEDEF KELİMELER VE KOTA DAĞILIMI
# ==========================================
TARGET_KEYWORDS_QUOTA = {
    # Veri & İş Zekası / Analitik
    "Data Analyst": 5,
    "Veri Analisti": 5,
    "Business Intelligence": 3,
    "İş Zekası": 3,
    "Business Analyst": 5,
    "İş Analisti": 5,
    "Junior Data Analyst": 5,
    "SQL": 5,
    "Data Modeling": 2,
    "Veri Modelleme": 2,
    "Data Warehouse": 2,
    "Veri Ambarı": 2,
    "Endüstri Mühendisi": 5,
    "Product Owner": 3,
    "Product Specialist": 3,
    
    # ERP, SAP & Yazılım
    "ERP Danışmanı": 3,
    "SAP Consultant": 3,
    "ABAP": 5,
    "S/4HANA": 3,
    
    # Tedarik Zinciri, Üretim & Pazarlama
    "Supply Chain": 5,
    "Tedarik Zinciri": 5,
    "Üretim": 5,
    "Production": 3,
    "Marketing": 5,
    "Pazarlama": 5,
    
    # Genç Yetenek & Yönetici Adaylığı
    "Yeni Mezun": 2,
    "Graduate": 2,
    "Management Trainee": 2,
    "Yönetici Adayı": 1
}

# ==========================================
# 3. VERİ MODELLERİ
# ==========================================
class JobListing(BaseModel):
    title: str
    company: str
    location: str
    job_link: str
    job_id: str

class JobMatchAnalysis(BaseModel):
    match_score: int
    is_disqualified: bool
    disqualification_reason: str
    suitability_category: str
    extracted_requirements: List[str]
    matching_points: List[str]
    missing_or_risk_points: List[str]
    brief_summary: str

# ==========================================
# 4. HAFIZA (DEDUPLICATION) YÖNETİMİ
# ==========================================
def load_processed_job_ids() -> set:
    """Daha önce incelenmiş ilan ID'lerini dosyadan okur."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("processed_ids", []))
        except Exception:
            return set()
    return set()

def save_processed_job_ids(processed_ids: set):
    """Güncel incelenmiş ilan ID listesini JSON olarak kaydeder."""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed_ids": list(processed_ids)}, f, indent=2)

# ==========================================
# 5. CV OKUMA VE SCRAPER FONKSİYONLARI
# ==========================================
def extract_text_from_pdf(pdf_path: str = "cv.pdf") -> str:
    """PDF formatındaki CV'den metin ayıklar."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"'{pdf_path}' bulunamadı! Lütfen GitHub reposuna 'cv.pdf' dosyasını yükleyin.")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def fetch_linkedin_jobs_by_url(url: str) -> List[JobListing]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    job_list = []
    try:
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code != 200:
            print(f"   ⚠️ LinkedIn HTTP Yanıtı Başarısız: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all("li")

        for card in job_cards:
            title_tag = card.find("h3", class_="base-search-card__title")
            company_tag = card.find("h4", class_="base-search-card__subtitle")
            location_tag = card.find("span", class_="job-search-card__location")
            link_tag = card.find("a", class_="base-card__full-link")

            if title_tag and company_tag and link_tag:
                link = link_tag.get("href", "").split("?")[0]
                job_list.append(JobListing(
                    title=title_tag.get_text(strip=True),
                    company=company_tag.get_text(strip=True),
                    location=location_tag.get_text(strip=True) if location_tag else "Belirtilmemiş",
                    job_link=link,
                    job_id=link.rstrip("/").split("-")[-1]
                ))
        return job_list
    except Exception as e:
        print(f"   [-] Scrape hatası: {e}")
        return []
        
def fetch_job_details(job_id: str) -> str:
    """İlanın detay açıklamasını çeker."""
    detail_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(detail_url, headers=headers, timeout=8)
        if response.status_code != 200:
            return "Detay metni çekilemedi."
        soup = BeautifulSoup(response.text, "html.parser")
        desc = soup.find("div", class_="show-more-less-html__markup")
        return desc.get_text(separator="\n", strip=True) if desc else "Açıklama bulunamadı."
    except Exception as e:
        return f"Hata: {e}"

# ==========================================
# 6. GEMINI PUANLAMA (YEDEKLİ MODEL YAPISI)
# ==========================================
def evaluate_job_with_gemini(job_title: str, company: str, raw_description: str, cv_text: str) -> JobMatchAnalysis:
    prompt = f"""
    Sen uzman bir İK ve Teknik Kariyer Danışmanısın.
    
    Aşağıdaki ilanı adayın CV'sine göre incele.

    📌 ADAYIN PROFİLİ VE TECRÜBE DEĞERLENDİRMESİ:
    - Adayın güçlü ve uzun dönemli staj tecrübeleri bulunmaktadır.
    - İlanlarda 1-2 yıl veya "en az 2-3 yıl" gibi giriş/orta seviye tecrübe beklentileri varsa, adayın bu şartı staj ve proje tecrübeleriyle karşılayabileceğini varsay ve KESİNLİKLE ELEME.
    
    🚨 KESİN ELEME KRİTERLERİ (Disqualification Rules):
    Aşağıdaki durumlardan BİRİ BİLE varsa, "is_disqualified": true, "match_score": 0 yap ve nedenini "disqualification_reason" alanına yaz:
    1. İlan 3 yıl veya daha fazla (3+, 4+, 5+ yıl vb.) zorunlu iş tecrübesi istiyorsa.
    2. Pozisyon Senior, Lead, Principal, Yönetici veya Direktör seviyesindeyse.
    3. İş tanımı saf yazılım mimarisi (örn. React, iOS, DevOps, Kubernetes) veya analitik olmayan rutin idari/ofis işiyse.
    
    Eğer elenme sebebi yoksa adayın yetkinlikleri ile ilanı 0-100 arasında puanla.
    
    MUTLAKA sadece aşağıdaki JSON formatında geçerli bir JSON çıktısı üret:
    {{
      "is_disqualified": false,
      "disqualification_reason": "Yok veya elenme sebebi",
      "match_score": 85,
      "suitability_category": "Yüksek Uyum",
      "extracted_requirements": ["SQL", "Power BI", "Süreç Analizi"],
      "matching_points": ["Endüstri Mühendisliği mezuniyeti", "SQL yetkinliği"],
      "missing_or_risk_points": ["İlgili sektörde staj tecrübesi tercihi"],
      "brief_summary": "Junior analist rolü olup adayın profiliyle yüksek uyum göstermektedir."
    }}

    ADAYIN CV METNİ:
    {cv_text}
    ---
    Pozisyon: {job_title} | Şirket: {company}
    İlan Metni: {raw_description}
    """

    fallback_models = ['gemini-flash-latest', 'gemini-3.7-flash', 'gemini-3.1-flash-lite']
    last_exception = None

    for model_name in fallback_models:
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                clean_json = response.text.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.replace("```json", "", 1)
                if clean_json.endswith("```"):
                    clean_json = clean_json.rstrip("```").strip()

                return JobMatchAnalysis.model_validate_json(clean_json)
            except Exception as e:
                last_exception = e
                time.sleep(2 * attempt)

    raise last_exception

# ==========================================
# 7. HEDEF KOTALI VE SAYFALAMALI TARAMA
# ==========================================
INSTANT_DISQUALIFY_KEYWORDS = [
    "senior", "sr.", "sr ", "lead", "principal", "director", "head of", 
    "müdür", "yönetici", "takım lideri", "chief", "executive", "intern", "internship" , "staj" , "stajyer"
]

def fetch_and_evaluate_target_jobs(keyword: str, target_new_count: int, cv_text: str):
    processed_ids = load_processed_job_ids()
    high_match_jobs = []
    analyzed_new_count = 0
    start_offset = 0
    max_pages = 4

    print(f"\n🔍 '{keyword}' için {target_new_count} adet YENİ ilan aranıyor...")

    while analyzed_new_count < target_new_count and max_pages > 0:
        encoded_keyword = quote_plus(keyword)
        url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={encoded_keyword}&location=Turkey&start={start_offset}"
        jobs_on_page = fetch_linkedin_jobs_by_url(url)
        
        # Sayfa kontrolü
        if not jobs_on_page:
            print(f"   ⚠️ Sayfa boş döndü veya çekilemedi (offset: {start_offset}). Sonraki anahtar kelimeye geçiliyor.")
            break

        print(f"   📄 Sayfa çekildi (offset: {start_offset}) -> Bulunan ilan sayısı: {len(jobs_on_page)}")

        for job in jobs_on_page:
            if analyzed_new_count >= target_new_count:
                break

            # 1. Kontrol: Hafızada var mı?
            if job.job_id in processed_ids:
                print(f"   ⏩ [Daha Önce İncelendi]: {job.title} ({job.company})")
                continue

            # 2. Kontrol: Senior/Müdür filtresi
            title_lower = job.title.lower()
            if any(bad_kw in title_lower for bad_kw in INSTANT_DISQUALIFY_KEYWORDS):
                print(f"   🚫 [Hızlı Elendi - Kıdem]: {job.title} ({job.company})")
                processed_ids.add(job.job_id)
                continue

            # 3. Kontrol: Analize gönder
            analyzed_new_count += 1
            print(f"   🤖 [{analyzed_new_count}/{target_new_count}] Analiz ediliyor: {job.title} - {job.company}")
            
            try:
                desc = fetch_job_details(job.job_id)
                analysis = evaluate_job_with_gemini(job.title, job.company, desc, cv_text)
                processed_ids.add(job.job_id)

                if analysis.is_disqualified:
                    print(f"      ❌ [AI Tarafından Elendi]: {analysis.disqualification_reason}")
                elif analysis.match_score >= SCORE_THRESHOLD:
                    print(f"      ⭐ Eşik Geçildi: {analysis.match_score}/100")
                    high_match_jobs.append({"job": job, "analysis": analysis})
                else:
                    print(f"      📊 Puan: {analysis.match_score}/100")
            except Exception as e:
                print(f"      [-] Analiz hatası: {e}")

            time.sleep(2)

        start_offset += 25
        max_pages -= 1

    if analyzed_new_count == 0:
        print(f"   ℹ️ '{keyword}' kategorisinde analiz edilecek yeni ilan bulunamadı.")

    save_processed_job_ids(processed_ids)
    return high_match_jobs

# ==========================================
# 8. RAPORLAMA VE MAİL GÖNDERİMİ
# ==========================================
def generate_email_html(matched_jobs: list) -> str:
    cards = ""
    for item in matched_jobs:
        job = item["job"]
        a = item["analysis"]
        cards += f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 6px; padding: 16px; margin-bottom: 16px; background: #fafafa;">
          <div style="font-size: 16px; font-weight: bold; color: #1e40af;">{job.title} <span style="background:#10b981; color:white; font-size:12px; padding:2px 8px; border-radius:10px; float:right;">{a.match_score}/100</span></div>
          <div style="color: #64748b; font-size: 13px; margin: 4px 0 10px 0;">🏢 {job.company} | 📍 {job.location}</div>
          <div style="font-size: 14px; color: #334155; margin-bottom: 8px;"><strong>📝 AI Özeti:</strong> {a.brief_summary}</div>
          <div style="font-size: 13px; color: #15803d;"><strong>✅ Güçlü Yönler:</strong> {", ".join(a.matching_points[:2])}</div>
          <div style="font-size: 13px; color: #b91c1c; margin-top: 4px;"><strong>⚠️ Riskler:</strong> {", ".join(a.missing_or_risk_points[:2])}</div>
          <a href="{job.job_link}" style="display:inline-block; margin-top:10px; background:#2563eb; color:white; text-decoration:none; padding:6px 12px; border-radius:4px; font-size:12px;" target="_blank">İlanı Aç →</a>
        </div>
        """
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; background: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px;">
          <h2 style="color: #0f172a; border-bottom: 2px solid #2563eb; padding-bottom: 10px;">🎯 Günlük İş İlanı Bülteni</h2>
          {cards}
          <p style="font-size: 11px; color: #94a3b8; text-align: center; margin-top: 20px;">CV Tabanlı Otomasyon Botu</p>
        </div>
      </body>
    </html>
    """

def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"İş Takip Botu <{SENDER_EMAIL}>"
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print("📧 E-posta başarıyla gönderildi!")

# ==========================================
# 9. ANA ÇALIŞTIRMA FONKSİYONU
# ==========================================
def main():
    print("📄 CV okunuyor...")
    cv_text = extract_text_from_pdf("cv.pdf")
    print(f"✅ CV başarıyla okundu ({len(cv_text)} karakter).")

    all_matched = []
    total_target = sum(TARGET_KEYWORDS_QUOTA.values())
    print(f"\n🚀 Toplam {len(TARGET_KEYWORDS_QUOTA)} kategori için {total_target} adet hedef ilan taraması başlatılıyor...\n")

    for keyword, quota in TARGET_KEYWORDS_QUOTA.items():
        matched = fetch_and_evaluate_target_jobs(
            keyword=keyword, 
            target_new_count=quota, 
            cv_text=cv_text
        )
        all_matched.extend(matched)

    if all_matched:
        all_matched.sort(key=lambda x: x["analysis"].match_score, reverse=True)
        html = generate_email_html(all_matched)
        send_email(f"🚀 Günün Eşleşen En İyi {len(all_matched)} İlanı", html)
    else:
        print("\n[-] Eşik puanı geçen yeni ilan bulunamadı.")

# ==========================================
# 10. GİRİŞ NOKTASI (ENTRYPOINT)
# ==========================================
if __name__ == "__main__":
    main()
