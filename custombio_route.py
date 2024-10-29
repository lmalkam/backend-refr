from fastapi import FastAPI, HTTPException, Query, APIRouter
from pydantic import BaseModel
import google.generativeai as genai
import requests
import os, re
from dotenv import load_dotenv

app = FastAPI()
BioRouter = APIRouter()

load_dotenv(".env.local")
GEMINI_API_KEY = os.getenv("API_KEY")
QUERY2 = os.getenv("QUERY2")

# Set up your Google Generative AI API key
genai.configure(api_key=GEMINI_API_KEY)

class BioResponse(BaseModel):
    bio: str

def extract_text_from_pdf(pdf_url: str) -> str:
    # Placeholder function; replace with PDF text extraction using a library like pdfminer.
    response = requests.get(pdf_url)
    pdf_text = response.text  # Simplified, replace with actual PDF parsing.
    return pdf_text

import re

def generate_bio_with_gemini(resume_text: str) -> str:
    if not resume_text or not isinstance(resume_text, str):
        return ""
    
    # Clean up the input text by removing special characters (optional)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', resume_text).strip()
    
    # Generate content with Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"{QUERY2}\n\n{clean_text}"
    response = model.generate_content(prompt)
    
    # Extract and clean up the bio text from the response
    bio = response.text if isinstance(response.text, str) else str(response)
    bio = re.sub(r'\n+', ' ', bio)          # Replace \n with space
    bio = re.sub(r'\* \*\*', '', bio)       # Remove * ** bullet formatting
    bio = re.sub(r'\s+', ' ', bio).strip()  # Remove any extra spaces
    
    return bio


@BioRouter.post("/generate_bio", response_model=BioResponse)
async def generate_bio(resume_url: str = Query(..., description="URL to the resume file")):
    try:
        resume_text = extract_text_from_pdf(resume_url)
        #print(resume_text)
        bio = generate_bio_with_gemini(resume_text)
        return BioResponse(bio=bio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(BioRouter)