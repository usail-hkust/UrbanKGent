import uvicorn
from multiprocessing import Process
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-host", type=str, default='0.0.0.0')
    parser.add_argument("-port", type=str, default=8000)
    parser.add_argument("-workers", type=int, default=2) # multi threads
    return parser.parse_args()

"""

curl -X POST "http://localhost:8000"
     -H 'Content-Type: application/json'
     -d '{"prompt": "Introduce yourself.", 
          "history": [("Helo", "Hi, How can I help you today?")], 
          "model": "llama-2-7b-chat-hf"}'
"""


if __name__ == '__main__':
    args = parse_args()
    uvicorn.run(app='open_llm_app:app', host=args.host, port=args.port, workers=args.workers)
