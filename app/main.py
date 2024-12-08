from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.investment_advisor_llm import process_query

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the homepage with a text input form for user queries.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def handle_query(request: Request, query: str = Form(...)):
    """
    Handle user queries, pass them to the LLM workflow, and return the response.
    """
    # Call the processing function
    response = process_query(query)
    # Render the response in the frontend
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user_query": query, "response": response}
    )