import json

def prompt_completion_transfer(prompt_completion):
    transfered_prompt_completion = {}

    transfered_prompt_completion['model'] = prompt_completion['model']
    transfered_prompt_completion['system_message'] = "You're a helpful assistant."
    transfered_prompt_completion['temperature'] =  prompt_completion['temperature']
    list_prompt = prompt_completion['messages']

    prompt = " "
    for index in range(len(list_prompt)):
        prompt = prompt + list_prompt[index]['content']

    transfered_prompt_completion['prompt'] = prompt
    transfered_prompt_completion['history'] = [("Hello", "Hi, How can I help you today?")]

    return transfered_prompt_completion
