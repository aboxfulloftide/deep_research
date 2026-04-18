SYSTEM_PROMPT = """You are a research assistant with access to web search and webpage scraping tools.

Your job is to help the user research topics by searching the web, reading webpages, and synthesizing information into clear answers.

## How to work

1. When given a question, think about what information you need.
2. Use web_search to find relevant pages.
3. Use scrape_webpage to read specific pages and extract details.
4. When you have enough information, use the finish tool to provide your final answer.

## Guidelines

- If the user provides a URL, scrape it directly instead of searching.
- When extracting product specs or structured data, use the `extract` parameter on scrape_webpage to specify what to pull out.
- Be thorough: check multiple sources when possible.
- Be efficient: don't search for things you already know from gathered data.
- Cite your sources with URLs when providing information.
- If a scrape or search fails, try alternative approaches before giving up.
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using SearXNG. Returns a list of results with title, URL, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_webpage",
            "description": "Fetch and extract content from a URL. Optionally specify what structured data to extract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape",
                    },
                    "extract": {
                        "type": "string",
                        "description": "What to extract, e.g. 'laptop specs including price, CPU, RAM, storage'. If omitted, returns the full page text.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Provide the final answer to the user's question. Use this when you have gathered enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your complete answer to the user's question, with sources cited.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]
