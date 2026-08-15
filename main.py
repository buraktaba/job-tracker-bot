import os
import time
import json
import smtplib
import requests
from bs4 import BeautifulSoup
from typing import List
from pydantic import BaseModel, Field
from pypdf import PdfReader
from google import genai
from google.genai import types
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# 1. Çevre Değişkenlerini Yükle
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", SENDER_EMAIL)
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "65"))

if not GEMINI_API_KEY or not SENDER_EMAIL or not EMAIL_PASSWORD:
    raise ValueError("Gerekli çevre değişkenleri (GEMINI_API_KEY, SENDER_EMAIL, EMAIL_PASSWORD) eksik!")

client = genai.Client(api_key=GEMINI_API_KEY)

# 2. PDF'ten CV Metnini Otomatik Okuma
def extract_text_from_pdf(pdf_path: str = "cv.pdf") -> str:
    """Repo içerisindeki cv.pdf dosyasını otomatik okur."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"'{pdf_path}' dosyası bulunamadı! Lütfen GitHub reposuna 'cv.pdf' dosyasını yükleyin.")
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# 3. Veri Modelleri
class JobListing(BaseModel):
    title: str
    company: str
    location: str
    job_link: str
    job_id: str

class JobMatchAnalysis(BaseModel):
    match_score: int
    suitability_category: str
    extracted_requirements: List[str]
    matching_points: List[str]
    missing_or_risk_points: List[str]
    brief_summary: str

# 4. Scraper Fonksiyonları
def fetch_linkedin_jobs(keyword: str, location: str = "Turkey", limit: int = 3) -> List[JobListing]:
    url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={keyword}&location={location}&start=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    job_list = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        job_cards = soup.find_all("li")

        for card in job_cards:
            if len(job_list) >= limit:
                break
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
        print(f"[-] Scrape hatası: {e}")
        return []

def fetch_job_details(job_id: str) -> str:
    detail_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    try:
        response = requests.get(detail_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return "Detay metni çekilemedi."
        soup = BeautifulSoup(response.text, "html.parser")
        desc = soup.find("div", class_="show-more-less-html__markup")
        return desc.get_text(separator="\n", strip=True) if desc else "Açıklama bulunamadı."
    except Exception as e:
        return f"Hata: {e}"

# 5. Gemini Puanlama (Dinamik CV ile)
def evaluate_job_with_gemini(job_title: str, company: str, raw_description: str, cv_text: str) -> JobMatchAnalysis:
    prompt = f"""
    Sen uzman bir İnsan Kaynakları ve Teknik Kariyer Danışmanısın.
    
    Aşağıda adayın gerçek özgeçmiş (CV) metni ve bir iş ilanının detayları yer almaktadır.
    
    GÖREVİN:
    1. İlan metnindeki kurumsal dolgu tanıtımları tamamen ele.
    2. İlanın aradığı yetkinlikler (teknik araçlar, sorumluluklar, tecrübe beklentisi) ile adayın CV'sindeki eğitim, projeler, teknik beceriler ve staj/iş deneyimlerini doğrudan kıyasla.
    3. CV ile ilan arasındaki uyumu 0-100 arasında objektif olarak puanla.
    
    MUTLAKA sadece aşağıdaki JSON formatında geçerli bir JSON çıktısı üret:
    {{
      "match_score": 80,
      "suitability_category": "Yüksek Uyum",
      "extracted_requirements": ["Python", "SQL", "Veri Modelleme"],
      "matching_points": ["CV'deki SQL deneyimi", "Endüstri Mühendisliği eğitimi"],
      "missing_or_risk_points": ["2 yıl deneyim beklentisi"],
      "brief_summary": "Pozisyon veri analitiği odaklı olup CV ile güçlü bir uyum göstermektedir."
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
                        temperature=0.2,
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
                print(f"  [!] {model_name} modeli geçici hata verdi ({attempt}/2): {e}")
                time.sleep(2 * attempt)
        print(f"  ↪️ {model_name} meşgul, yedek modele geçiliyor...")

    raise last_exception

# 6. Mail Gönderimi
def generate_email_html(high_match_jobs: list) -> str:
    cards = ""
    for item in high_match_jobs:
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
          <p style="font-size: 11px; color: #94a3b8; text-align: center; margin-top: 20px;">CV Tabanlı Otomasyon Botu Tarafından Gönderildi.</p>
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

# 7. Ana Akış
def main():
    print("📄 CV okunuyor...")
    cv_text = extract_text_from_pdf("cv.pdf")
    print(f"✅ CV başarıyla okundu ({len(cv_text)} karakter).")

    target_keywords = ["Data Analyst", "Endüstri Mühendisi"]
    all_matched = []

    for kw in target_keywords:
        print(f"\n🔍 '{kw}' aranıyor...")
        jobs = fetch_linkedin_jobs(keyword=kw, location="Istanbul, Turkey", limit=3)
        for job in jobs:
            print(f"🤖 Analiz ediliyor: {job.title} ({job.company})")
            try:
                desc = fetch_job_details(job.job_id)
                analysis = evaluate_job_with_gemini(job.title, job.company, desc, cv_text)
                
                print(f"  📊 Uyum Puanı: {analysis.match_score}/100")
                if analysis.match_score >= SCORE_THRESHOLD:
                    print(f"  ⭐ Eşik Geçildi ({analysis.match_score} >= {SCORE_THRESHOLD})")
                    all_matched.append({"job": job, "analysis": analysis})
            except Exception as err:
                print(f"  [-] Analiz hatası: {err}")
            time.sleep(1)

    if all_matched:
        all_matched.sort(key=lambda x: x["analysis"].match_score, reverse=True)
        html = generate_email_html(all_matched)
        send_email(f"🚀 Günün Eşleşen İlanları ({len(all_matched)} Fırsat)", html)
    else:
        print("\n[-] Eşik puanı geçen yeni ilan bulunamadı.")

if __name__ == "__main__":
    main()
