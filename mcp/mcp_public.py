from typing import Any
from mcp.server.fastmcp import FastMCP
import http.client
import json

mindverse = FastMCP("mindverse_public")
url = "app.secondme.io"

messages =[]

@mindverse.tool()
async def get_response(query:str, instance_id:str) -> str | None:
    """
    Received a response based on public secondme model.

    Args:
        query (str): Questions raised by users regarding the secondme model.
        instance_id (str): ID used to identify the secondme model, or url used to identify the secondme model.

    """
    id = instance_id.split('/')[-1]
    path = f"/api/chat/{id}"
    headers = {"Content-Type": "application/json"}
    messages.append({"role": "user", "content": query})

    data = {
        "messages": messages,
        "metadata": {
        "enable_l0_retrieval": False,
        "role_id": "default_role"
    },
    "temperature": 0.7,
    "max_tokens": 2000,
    "stream": True
    }

    conn = http.client.HTTPSConnection(url)

    # Send the POST request
    conn.request("POST", path, body=json.dumps(data), headers=headers)

    # Get the response
    response = conn.getresponse()

    full_content = ""

    for line in response:
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line == 'data: [DONE]':
                break
            if decoded_line.startswith('data: '):
                try:
                    json_str = decoded_line[6:]
                    chunk = json.loads(json_str)
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        full_content += content
                except json.JSONDecodeError:
                    pass

    conn.close()
    if full_content:
        messages.append({"role": "system", "content": full_content})
        return full_content
    else:
        return None

if __name__ == "__main__":

    mindverse.run(transport='stdio')



