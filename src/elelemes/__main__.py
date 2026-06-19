import asyncio

# import aioconsole as aioc
import prompt_toolkit as ptk
import ollama

async def read_input(input_queue: asyncio.Queue, prompt_session: ptk.PromptSession):
    while True:
        with ptk.patch_stdout.patch_stdout():
            msg = await prompt_session.prompt_async("> ")
        await input_queue.put(msg)

async def main():
    ptk.print_formatted_text("Hello world!")

    input_queue = asyncio.Queue()
    prompt_session = ptk.PromptSession()
    asyncio.create_task(read_input(input_queue, prompt_session))

    llm_msgs = []
    ollama_cl = ollama.AsyncClient()

    while True:
        msg = await input_queue.get()
        # await asyncio.sleep(0.5)
        # print(msg)
        llm_msgs.append({'role': 'user', 'content': msg})

        ptk.print_formatted_text("Generating...")
        resp_stream = await ollama_cl.chat(
            model='gemma4:e4b',
            messages=llm_msgs,
            stream=True)
        prev_status = ''
        full_thinking = ""
        full_resp = ""
        tool_calls_reqs = []
        async for chunk in resp_stream:
            thinking = chunk.message.thinking
            if thinking:
                if prev_status != 'think':
                    ptk.print_formatted_text("\n## Thinks:")
                    prev_status = 'think'
                ptk.print_formatted_text(thinking, end='', flush=True)
                full_thinking += thinking

            cont = chunk.message.content
            if cont:
                if prev_status != 'response':
                    ptk.print_formatted_text("\n## Responds:")
                    prev_status = 'response'
                ptk.print_formatted_text(cont, end='', flush=True)
                full_resp += cont

            tool_calls = chunk.message.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    ptk.print_formatted_text(f"\n## Calls tool {tc.function.name}")
                tool_calls_reqs.extend(tool_calls)

        ptk.print_formatted_text('')
        
        assistant_msg = {'role': 'assistant', 'content': full_resp}
        if full_thinking:
            assistant_msg['thinking'] = full_thinking
        if tool_calls:
            assistant_msg['tool_calls'] = tool_calls_reqs
        llm_msgs.append(assistant_msg)
        
        if len(tool_calls_reqs) > 0:
            for tc in tool_calls:
                t_name = tc.function.name
                t_args = tc.function.arguments
                ptk.print_formatted_text(f"n## Executing tool {t_name} with args {t_args}")

if __name__ == '__main__':
    asyncio.run(main())
