from fastapi import FastAPI
from pydantic import BaseModel
from app.query_engine import answer_query

app = FastAPI()

# Root route
@app.get("/")
def home():
    return {"message": "CodeGraph backend running"}

# Request body structure
class RepoRequest(BaseModel):
    repo_url: str
    query: str = "explain the overall architecture of this codebase"

# Analyze repo endpoint
@app.post("/analyze-repo")
def analyze_repo(request: RepoRequest):
    try:
        result = answer_query(request.repo_url, request.query)
        return {"analysis": result}
    except Exception as e:
        return {"error": str(e)}