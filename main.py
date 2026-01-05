from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import io
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import psycopg2
from psycopg2.extras import RealDictCursor
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests


# Load environment variables
load_dotenv()

# Try to import libraries
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://YOUR-VERCEL-APP.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database connection
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

# Initialize database tables
def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                password_hash VARCHAR(255),
                google_id VARCHAR(255) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Pydantic models
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    token: str
    user: dict

class QuestionRequest(BaseModel):
    resume_text: str
    difficulty: str = "medium"
    question_type: str = "technical"  # technical or behavioral

# Helper functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# File parsing functions
def extract_text_from_pdf(file_content: bytes) -> str:
    if PyPDF2 is None:
        raise HTTPException(status_code=500, detail="PyPDF2 not installed")
    
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file_content: bytes) -> str:
    if docx is None:
        raise HTTPException(status_code=500, detail="python-docx not installed")
    
    doc = docx.Document(io.BytesIO(file_content))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def parse_resume(file_content: bytes, filename: str) -> str:
    if filename.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif filename.lower().endswith('.docx'):
        return extract_text_from_docx(file_content)
    elif filename.lower().endswith('.txt'):
        return file_content.decode('utf-8')
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

def generate_questions_with_gemini(resume_text: str, difficulty: str, question_type: str) -> List[dict]:
    if not GEMINI_AVAILABLE:
        print("Warning: Gemini not available. Using mock questions.")
        return get_mock_questions(difficulty, question_type)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Using mock questions.")
        return get_mock_questions(difficulty, question_type)
    
    try:
        genai.configure(api_key=api_key)
        
        prompt = f"""You are an expert technical interviewer and career coach. Based on the resume below, generate  personalized interview questions that collectively prepare the candidate for common interview rounds (phone screen, technical coding, system-design/architecture, domain-role fit, manager/leadership round, and HR/culture fit). Use the variable values provided below.

Resume:
{resume_text[:2000]}

Parameters (use these variables exactly):
- Question Type: {question_type}
- Difficulty Level: {difficulty}

Instructions (must be followed exactly):
1. Produce a **mix** that always includes:
   - HR/behavioral or culture-fit questions (label these categories as "HR - Behavioral" or "HR - Culture/Fit").
   - technical questions (mix of "Technical - Coding", "Technical - System Design", or "Technical - Concepts" as appropriate to the candidate's experience).
   - role/domain-fit question (label: "Domain/Role Fit" or "Technical - Concepts").
   - Optionally 1 aptitude/puzzle or on-the-spot thinking question (label: "Aptitude/Puzzle") where useful.

2. Make every question directly tied to information in the resume: reference project names, technologies, metrics, responsibilities, timelines, or companies when present. If the resume lacks necessary detail for a specific technical question, generate a **resume-clarifying** question labeled "HR - CV Clarification" that asks for the missing detail.

3. For behavioral questions, require STAR-structured follow-ups: the `followup` field should prompt the candidate to describe Situation, Task, Action, Result (or list 2–3 bullet prompts such as "Describe the situation, your specific actions, quantifiable outcome").

4. For technical questions, the `followup` field should include targeted follow-ups or hints the interviewer would ask (for example: "Ask for algorithm complexity, tradeoffs, edge cases, and an example test case" or "Ask them to draw architecture and explain component interactions and scaling").

5. Each question must be specific, actionable, and interview-ready (not generic). Avoid vague prompts like "Tell me about yourself" — prefer "Explain your role in X project and the technical decisions you made."

6. Output format requirement (strict): Return **ONLY** a valid JSON array with this exact structure (no markdown, no code blocks, no extra text). Use the `{difficulty}` value verbatim for every question's `difficulty`. Example structure to follow exactly:

[
  {{
    "question": "Your personalized question here based on the resume",
    "category": "Category name (e.g., Technical - Coding, HR - Behavioral, Technical - System Design, HR - CV Clarification, Domain/Role Fit, Aptitude/Puzzle)",
    "difficulty": "{difficulty}",
    "followup": "Optional follow-up question or interviewer prompts (for behavioral include STAR prompts; for technical include what to probe: complexity, tradeoffs, test cases)"
  }}
]

7. Choose categories only from this controlled list: "Technical - Coding", "Technical - System Design", "Technical - Concepts", "Domain/Role Fit", "HR - Behavioral", "HR - Culture/Fit", "HR - CV Clarification", "Aptitude/Puzzle". Use one category per question.

8. Tone: professional interviewer voice. Do not add scoring, sample answers, or any additional fields. Do not include clarifying questions to the user—if the resume is short, produce clarifying CV questions as required.

IMPORTANT: If the input `question_type` implies a sub-type (e.g., "coding", "behavioral", "full"), bias the selection to satisfy that request while still obeying the mix and round-coverage rules above.

Now generate the JSON array of questions using the resume and the parameter values provided.
"""


        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join([line for line in lines if not line.startswith("```")])
            response_text = response_text.strip()
        
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()
        
        questions = json.loads(response_text)
        print(f"✅ Generated {len(questions)} AI questions")
        return questions
        
    except Exception as e:
        print(f"Error calling Gemini: {str(e)}")
        return get_mock_questions(difficulty, question_type)

def get_mock_questions(difficulty: str, question_type: str) -> List[dict]:
    """Fallback mock questions based on type"""
    
    if question_type == "technical":
        return [
            {
                "question": "Based on your experience with [technology from resume], explain how you would approach building a scalable solution.",
                "category": "System Design",
                "difficulty": difficulty,
                "followup": "What trade-offs would you consider?"
            },
            {
                "question": "Looking at your projects, describe a challenging technical problem you solved and your approach.",
                "category": "Problem Solving",
                "difficulty": difficulty,
                "followup": "How did you measure the success of your solution?"
            },
            {
                "question": "Explain the most complex algorithm or data structure you've implemented in your experience.",
                "category": "Technical Knowledge",
                "difficulty": difficulty,
                "followup": "Why did you choose that particular approach?"
            }
        ]
    else:  # behavioral
        return [
            {
                "question": "Tell me about a time when you had to work with a difficult team member. How did you handle it?",
                "category": "Teamwork",
                "difficulty": difficulty,
                "followup": "What was the outcome and what would you do differently?"
            },
            {
                "question": "Describe a situation where you had to meet a tight deadline. How did you prioritize your tasks?",
                "category": "Time Management",
                "difficulty": difficulty,
                "followup": "What did you learn from that experience?"
            },
            {
                "question": "Give an example of when you showed leadership, even if you weren't in a leadership position.",
                "category": "Leadership",
                "difficulty": difficulty,
                "followup": "How did others respond to your leadership?"
            }
        ]

# Routes
@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/")
async def root():
    return {
        "message": "AI Interview Generator API with Authentication",
        "status": "running",
        "ai": "Gemini"
    }

@app.post("/api/auth/signup")
async def signup(user: UserSignup):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if user exists
        cur.execute("SELECT id FROM users WHERE email = %s", (user.email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password and create user
        password_hash = hash_password(user.password)
        cur.execute(
            "INSERT INTO users (email, name, password_hash) VALUES (%s, %s, %s) RETURNING id",
            (user.email, user.name, password_hash)
        )
        user_id = cur.fetchone()['id']
        conn.commit()
        
        # Create token
        token = create_access_token({"user_id": user_id})
        
        cur.close()
        conn.close()
        
        return {
            "token": token,
            "user": {"id": user_id, "email": user.email, "name": user.name}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login")
async def login(user: UserLogin):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Find user
        cur.execute("SELECT id, email, name, password_hash FROM users WHERE email = %s", (user.email,))
        db_user = cur.fetchone()
        
        if not db_user or not verify_password(user.password, db_user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Create token
        token = create_access_token({"user_id": db_user['id']})
        
        cur.close()
        conn.close()
        
        return {
            "token": token,
            "user": {"id": db_user['id'], "email": db_user['email'], "name": db_user['name']}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/google")
async def google_login():
    """Redirect to Google OAuth"""
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")
    
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured. Please set GOOGLE_CLIENT_ID in .env")
    
    print(f"🔵 Google OAuth initiated")
    print(f"   Client ID: {GOOGLE_CLIENT_ID[:20]}...")
    print(f"   Redirect URI: {REDIRECT_URI}")
    
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=openid email profile&"
        f"access_type=offline&"
        f"prompt=consent"
    )
    
    print(f"   Redirecting to: {google_auth_url[:100]}...")
    
    return RedirectResponse(url=google_auth_url)

@app.get("/api/auth/google/callback")
async def google_callback(code: str):
    """Handle Google OAuth callback"""
    try:
        import requests as req
        
        GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")
        FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
        
        # Exchange code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        
        token_response = req.post(token_url, data=token_data)
        token_json = token_response.json()
        
        if "error" in token_json:
            raise HTTPException(status_code=400, detail=token_json["error"])
        
        # Verify ID token
        id_info = id_token.verify_oauth2_token(
            token_json["id_token"],
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
        email = id_info["email"]
        name = id_info.get("name", email.split("@")[0])
        google_id = id_info["sub"]
        
        # Check if user exists or create new user
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT id, email, name FROM users WHERE email = %s OR google_id = %s", (email, google_id))
        user = cur.fetchone()
        
        if user:
            user_id = user["id"]
            # Update google_id if not set
            cur.execute("UPDATE users SET google_id = %s WHERE id = %s", (google_id, user_id))
        else:
            # Create new user
            cur.execute(
                "INSERT INTO users (email, name, google_id) VALUES (%s, %s, %s) RETURNING id",
                (email, name, google_id)
            )
            user_id = cur.fetchone()["id"]
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Create JWT token
        token = create_access_token({"user_id": user_id})
        
        # Redirect to frontend with token
        return RedirectResponse(url=f"{FRONTEND_URL}?token={token}&email={email}&name={name}")
        
    except Exception as e:
        print(f"Google OAuth error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_resume(file: UploadFile = File(...), user_id: int = Depends(get_current_user)):
    try:
        content = await file.read()
        resume_text = parse_resume(content, file.filename)
        
        return {
            "success": True,
            "filename": file.filename,
            "text_preview": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "text_length": len(resume_text)
        }
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-questions")
async def generate_questions(request: QuestionRequest, user_id: int = Depends(get_current_user)):
    try:
        print(f"User {user_id} generating questions: {request.question_type}, {request.difficulty}")
        
        questions = generate_questions_with_gemini(
            request.resume_text,
            request.difficulty,
            request.question_type
        )
        
        return {
            "success": True,
            "questions": questions,
            "count": len(questions)
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)