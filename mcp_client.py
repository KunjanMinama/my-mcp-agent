import subprocess, json, sys, threading, time
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Dict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="chat-completion",
        max_new_tokens=256,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
    )

# Pass llm to ChatHuggingFace
client = ChatHuggingFace(llm=llm)

server = subprocess.Popen(
    [sys.executable, "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    busize=1
)

print("🔗 Connected to MCP server")

def read_startup():
    while True:
        line = server.stderr.readline()
        if not line:
            break
        if "Starting MCP server" in line:
            break

startup_thread = threading.Thread(target=read_startup, daemon=True)
startup_thread.start()
time.sleep(1)


# MCP Communication Helpers
# We define functions to send requests and notifications over the MCP JSON-RPC protocol:
def send_request(method: str, params: Dict = None):
    global request_id
    request_id += 1
    req = {"jsonrpc": "2.0", "id": str(request_id), "method": method}
    if params:
        req["params"] = params
    server.stdin.write(json.dumps(req) + "\n")
    server.stdin.flush()
    return server.stdout.readline().strip()

def send_notification(method: str, params:Dict = None):
    req = {"jsonrpc": "2.0", "method": method}
    if params:
        req["params"] = params
    server.stdin.write(json.dumps(req) + "\n")
    server.stdin.flush()


# Initia;ize the MCP Connection

init_response = send_request("initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "mcp-client", "version": "1.0.0"}

})
send_notification("initialized")

## Calling MCP tools
#This function allows the agent to call any tool registered in the MCP server with parameters.

def call_mcp_tool(tool: str, args: Dict):
    global request_id
    request_id += 1
    req = {
        "jsonrpc": "2.0",
        "id": str(request_id),
        "method": "tools/call",
        "params": {"name": tool, "arguments": {"input": args}},
    }
    server.stdin.write(json.dumps(req) + "\n")
    server.stdin.flush()
    response = server.stdout.readline().strip()
    try:
        resp_data = json.loads(response)
        return resp_data.get("result") or f"Error: {resp_data.get('error')}"
    except:
        return response
    
## Agent State Graph

class S(TypedDict):
    msg: str
    tool_result: Any
    result: str
    