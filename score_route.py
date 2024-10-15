from typing import Dict, Any
from fastapi import FastAPI, APIRouter
import pdfplumber
from bs4 import BeautifulSoup
import random, re, time, requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import google.generativeai as genai
from dotenv import load_dotenv
import os
from io import BytesIO
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(".env.local")
GEMINI_API_KEY = os.getenv("API_KEY")
QUERY = os.getenv("QUERY")
PREFIX = os.getenv("PREFIX")
genai.configure(api_key=GEMINI_API_KEY)
app = FastAPI()
score_router = APIRouter()

def create_session_with_retries():
    session = requests.Session()
    retries = Retry(total=5,
                    backoff_factor=0.1,
                    status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36"
    ]
    return random.choice(user_agents)

def scrape_linkedin_job(url, max_retries=5):
    session = create_session_with_retries()

    for attempt in range(max_retries):
        try:
            headers = {"User-Agent": get_random_user_agent()}
            time.sleep(random.uniform(5, 10))  # Increased delay

            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            job_data = extract_job_data(soup)
            return job_data

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Unable to scrape the job data.")
                return None

            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)

def extract_job_data(soup):
    job_data = {}

    try:
        job_data["company"] = soup.find("div", {"class": "top-card-layout__card"}).find("a").find("img").get('alt')
    except AttributeError:
        job_data["company"] = None

    try:
        job_data["job_title"] = soup.find("div", {"class": "top-card-layout__entity-info"}).find("h1").text.strip()
    except AttributeError:
        job_data["job_title"] = None

    try:
        job_data["location"] = soup.find("span", {"class": "topcard__flavor--bullet"}).text.strip()
    except AttributeError:
        job_data["location"] = None

    try:
        criteria_list = soup.find("ul", {"class": "description__job-criteria-list"})
        criteria_items = criteria_list.find_all("li")
        for item in criteria_items:
            criterion = item.find("h3").text.strip().lower().replace(" ", "_")
            value = item.find("span").text.strip()
            job_data[criterion] = value
    except AttributeError:
        pass

    try:
        job_data["description"] = soup.find("div", {"class": "show-more-less-html__markup"}).text.strip()
    except AttributeError:
        job_data["description"] = None

    return job_data

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    with pdfplumber.open(BytesIO(pdf_content)) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

def process_linkedin_url(url):
    if "linkedin.com/jobs/collections" in url:
        match = re.search(r'currentJobId=(\d+)', url)
        if match:
            job_id = match.group(1)
            return f"https://www.linkedin.com/jobs/view/{job_id}"
    return url

def get_skills(text: str) -> list:
    """
    Extracts skills from a given text more efficiently using the generative AI model.
    Preprocesses text before sending the query and handles response carefully.
    """
    if not text or not isinstance(text, str):
        return []

    # Preprocess text by trimming and limiting noise (you could do more here)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
    
    # If there's nothing to query, return an empty list
    if not clean_text:
        return []
    
    try:
        # Query generative model only when clean text is available
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{QUERY} {clean_text}"
        response = model.generate_content(prompt)

        # Ensure response is valid
        if not response or not hasattr(response, 'text'):
            return []

        # Process response into skills
        skills = response.text.split(',')
        return [skill.strip().lower() for skill in skills if skill.strip()]
    
    except Exception as e:
        print(f"Error in get_skills: {e}")
        return []



class ScoreRequest(BaseModel):
    resume_url: str
    jobLink: str

@score_router.post('/calculatescore')
async def upload_resume(data: ScoreRequest) -> Dict[str, Any]:
    # Download the resume PDF from the provided URL
    try:
        response = requests.get(PREFIX + data.resume_url)
        response.raise_for_status()
        resume_content = response.content
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download the resume from the provided URL: {str(e)}"}

    

    # Extract text from the downloaded PDF
    resume_text = extract_text_from_pdf(resume_content)

    # Process the LinkedIn job URL
    processedLink = process_linkedin_url(data.jobLink)
    job_data = scrape_linkedin_job(processedLink)
    job_description = job_data.get('description', '')

    # Extract relevant skills
    relevant_skills = get_skills(job_description)
    resume_skills = get_skills(resume_text)

    if not resume_skills or not relevant_skills:
        print("One of the skill sets is empty.")
        return {
            "resume_skills": resume_skills,
            "job_skills": relevant_skills,
            "similarity_score": 0.0
        }
    
    print(job_data.get("job_title"))
    print(job_data.get("company"))
    print(job_data.get("location"))

    # Join skills into a single string
    text = [" ".join(resume_skills), " ".join(relevant_skills)]

    # Initialize CountVectorizer and fit_transform it to the text data
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)

    # Convert the count matrix into an array
    count_matrix_array = count_matrix.toarray()

    # Calculate cosine similarity between the two vectors
    cosine_sim = cosine_similarity(count_matrix_array)

    # The similarity score between the resume and job is in cosine_sim[0, 1]
    score = cosine_sim[0, 1] * 100  # Convert to percentage

    return {"compatibility_score": min(score + 20, 98) , "title": job_data.get("job_title"), "company" : job_data.get("company"), "location" : job_data.get("location"), "description" : job_data.get("description")}


app.include_router(score_router)