import os
import time
import smtplib
import requests
from bs4 import BeautifulSoup
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# 1. Çevre Değişkenlerini Yükle (.env veya GitHub Secrets)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL", SENDER_EMAIL)
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "65"))

if not GEMINI_API_KEY or not SENDER_EMAIL or not EMAIL_PASSWORD:
    raise ValueError("Gerekli çevre değişkenleri (GEMINI_API_KEY, SENDER_EMAIL, EMAIL_PASSWORD) eksik!")

# 2. Gemini İstemcisi
client = genai.Client(api_key=GEMINI_API_KEY)

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
def fetch_linkedin_jobs(keyword: str, location: str = "Turkey", limit: int = 5) -> List[JobListing]:
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

# 5. Gemini Puanlama
def evaluate_job_with_gemini(job_title: str, company: str, raw_description: str) -> JobMatchAnalysis:
    candidate_profile = """
    ADAY PROFİLİ:
    - Eğitim: Endüstri Mühendisliği Lisans Mezunu.
    - Seviye: Junior / Uzman Yardımcısı / Giriş Seviyesi.
    - Yetkinlik Alanları: Veri Analitiği (Python, SQL, BI), Tedarik Zinciri, Üretim Planlama & Süreç Optimizasyonu.
    - Hedef: Veri odaklı karar destek veya mühendislik/süreç/tedarik zinciri analitiği rolleri.
    """
    prompt = f"""
    Sen uzman bir İK ve Teknik Kariyer Danışmanısın.
    İlanın aradığı teknik/sosyal gereksinimleri, tecrübe beklentisini ve adayın profilini değerlendirip 0-100 arasında puanla.
    Kurumsal şirket tanıtımlarını yok say.
    
    {candidate_profile}
    ---
    Pozisyon: {job_title} | Şirket: {company}
    İlan Metni: {raw_description}
    """

    response = client.models.generate_content(
        model='gemini-flash-latest',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobMatchAnalysis,
            temperature=0.2,
        ),
    )
    return JobMatchAnalysis.model_validate_json(response.text)

# 6. Mail Şablonu ve Gönderim
def generate_email_html(high_match_jobs: list) -> str:
    cards = ""
    for item in high_match_jobs:
        job = item["job"]
        a = item["analysis"]
        cards += f"""
        
          {job.title} {a.match_score}/100
          🏢 {job.company} | 📍 {job.location}
          📝 AI Özeti: {a.brief_summary}
          ✅ Güçlü Yönler: {", ".join(a.matching_points[:2])}
          ⚠️ Riskler: {", ".join(a.missing_or_risk_points[:2])}
          İlanı Aç →
        
        """
    
    return f"""
    
      
        
          🎯 Günlük İş İlanı Bülteni
          {cards}
          GitHub Actions Otomasyon Botu Tarafından Gönderildi.
        
      
    
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
    target_keywords = ["Data Analyst", "Endüstri Mühendisi", "Tedarik Zinciri"]
    all_matched = []

    for kw in target_keywords:
        print(f"🔍 '{kw}' aranıyor...")
        jobs = fetch_linkedin_jobs(keyword=kw, location="Istanbul, Turkey", limit=3)
        for job in jobs:
            print(f"🤖 Analiz ediliyor: {job.title} ({job.company})")
            desc = fetch_job_details(job.job_id)
            analysis = evaluate_job_with_gemini(job.title, job.company, desc)
            
            if analysis.match_score >= SCORE_THRESHOLD:
                print(f"  ⭐ Eşik Puan Geçildi: {analysis.match_score}/100")
                all_matched.append({"job": job, "analysis": analysis})
            time.sleep(1)

    if all_matched:
        all_matched.sort(key=lambda x: x["analysis"].match_score, reverse=True)
        html = generate_email_html(all_matched)
        send_email(f"🚀 Günün Eşleşen İlanları ({len(all_matched)} Fırsat)", html)
    else:
        print("[-] Eşik puanı geçen yeni ilan bulunamadı.")

if __name__ == "__main__":
    main()
