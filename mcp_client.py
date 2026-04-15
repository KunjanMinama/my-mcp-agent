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
    bufsize=1
)

print("🔗 Connected to MCP server")

def read_startup():
    for _ in range(20):  # max wait
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
request_id = 0
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


# Initialize the MCP Connection

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
    if not response:
        return "NO response from MCP server"
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


# Implement Agent logic 
# 1. Routing Requests to Tools
#----------------AGENT LOGIC-------------------------------------------
def route_request(state: S):
    """Use LLM to determine which tool to use and extract parameters"""

    routing_prompt = f"""
You MUST respond in valid JSON only.

User request: {state["msg"]}

Available tools:
- get_weather (needs: city)
- web_search (needs: query)

Return ONLY this format:
{{
  "tool": "get_weather" | "web_search" | "none",
  "parameters": {{...}} or null
}}

Only use tools if clarity needed, For general conversation, use "none"."""
    
    response = client.invoke(routing_prompt)
# The agent use LLM to decide which tool is appropriate.
    try:
        routing_decision = json.loads(response.content)
        tool = routing_decision.get("tool")
        params = routing_decision.get("parameters")

        print(f"🤖 Routing decision: {routing_decision.get('reasoning')}")

        if tool == "get_weather" and params:
            result = call_mcp_tool("get_weather", params)
            return {"msg": state["msg"], "tool_result": result, "result": ""}

        elif tool == "web_search" and params:
            result = call_mcp_tool("web_search", params)
            return {"msg": state["msg"], "tool_result": result, "result": ""}
        
        else:
            return {"msg": state["msg"], "tool_result": None}
        
    except Exception as e:
        print(f"⚠️ Routing error: {e}")
        return {"msg": state["msg"], "tool_result": None}
    
# If no tool is needed, the agent continues with general conversation.

## 2. General Natural Language Responses

def generate_response(state: S):
    """Use LLM to generate a natural language response"""
    if state.get("tool_result") is None:
        # No tool was used, direct LLM response
        response = client.invoke(state["msg"])
        return {
            "msg": state["msg"],
            "tool_result": state.get("tool_result"),
            "result": response.content
        }
    
    # Format tool result for LLM
    tool_data = json.dumps(state["tool_result"], indent=2)
    
    prompt = f"""
User question: {state['msg']}

Tool results:
{tool_data}

Give a helpful answer in under 100 words.
"""


    response = client.invoke(prompt
    )
    
    return {
        "msg": state["msg"],
        "tool_result": state["tool_result"],
        "result": response.content
    }


## Building the LangGraph Workflow

g = StateGraph(S)
g.add_node("route", route_request)
g.add_node("respond", generate_response)
g.set_entry_point("route")
g.add_edge("route", "respond")
g.add_edge("respond", END)
graph= g.compile()

# ---------------- SAVE GRAPH VISUALIZATION ----------------
try:
    # Generate and save the graph as PNG
    png_data = graph.get_graph().draw_mermaid_png()
    with open("langgraph_diagram.png", "wb") as f:
        f.write(png_data)
    print("📊 Graph visualization saved as 'langgraph_diagram.png'\n")
except Exception as e:
    print(f"⚠️ Could not generate graph visualization: {e}")
    print("Note: Install graphviz system dependency if needed: brew install graphviz\n")


#Tsting the Agent

tests = [
    "What's the weather in Mumbai?",
    "Search for latest news about AI",
]

for t in tests:
    result = graph.invoke({"msg": t})
    print(f"AGENT: {result['result']}\n")