import asyncio

# import aioconsole as aioc
import prompt_toolkit as ptk
import ollama

async def read_input(input_queue: asyncio.Queue, input_evt: asyncio.Event, prompt_session: ptk.PromptSession):
    while True:
        await input_evt.wait()
        input_evt.clear()
        with ptk.patch_stdout.patch_stdout():
            msg = await prompt_session.prompt_async("> ")
        await input_queue.put(msg)

async def main():
    ptk.print_formatted_text("Hello world!")

    input_queue = asyncio.Queue()
    input_evt = asyncio.Event()
    input_evt.set()
    prompt_session = ptk.PromptSession()
    asyncio.create_task(read_input(input_queue, input_evt, prompt_session))

    llm_msgs = []
    ollama_cl = ollama.AsyncClient()

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'bc',
                'description': 'Run `bc` - a standard arbitrary precision calculator commonly found on Unix-like systems.',
                'parameters': {
                    'type': 'object',
                    'required': ['expr'],
                    'properties': {
                        'expr': {'type': 'string', 'description': 'Input for `bc` executable'}
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'python_exec',
                'description': 'Run the Python 3.12 interpreter. It will be executed in an isolated container. The program may not expect data on stdin (i.e. call `input`). The results should be printed to stdout. This tool will return an object with two fields: `stdout` and `stderr` - two strings containing the scripts\'s stdout and stderr, respectively, which are expected to be UTF-8 encoded.',
                'parameters': {
                    'type': 'object',
                    'required': ['code'],
                    'properties': {
                        'code': {'type': 'string', 'description': 'The Python source code to execute'}
                    }
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'ed_editor',
                'description': 'Run the Ed editor. It is a line-based editor originating from Unix, whose commonly used derivative, `sed`, is often used in shell scripts. The editor will receive editing instructions provided as parameter to this tool and perform specified operations. Everything printed by Ed will be returned. If modifications should be saved, an appropriate command must be added at the end of instructions. Common commands include: `i` for insert before, `a` for insert after, `w` for write, `p` for print, `n` for print with line number annotations. Commands might be preceded with a line selector, like number or `%` (entire buffer).',
                'parameters': {
                    'type': 'object',
                    'required': ['instructions'],
                    'properties': {
                        'file': {'type': 'string', 'description': 'File to work on. This parameter might be ommited to start with empty buffer'},
                        'instructions': {'type': 'string', 'description': 'The instructions for Ed to execute. If the file should be saved, a write instruction should be added here.'}
                    }
                }
            }
        }
    ]

    new_user_msg = True
    while True:
        if new_user_msg:
            msg = await input_queue.get()
            # await asyncio.sleep(0.5)
            # print(msg)
            llm_msgs.append({'role': 'user', 'content': msg})

            new_user_msg = False

        ptk.print_formatted_text("Generating...")
        ptk.print_formatted_text(f"{repr(llm_msgs)}")
        resp_stream = await ollama_cl.chat(
            model='gemma4:e4b',
            messages=llm_msgs,
            tools=tools,
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

            if chunk.done:
                ptk.print_formatted_text("\n## Done!")
                new_user_msg = True

        ptk.print_formatted_text('')
        
        assistant_msg = {'role': 'assistant', 'content': full_resp}
        if full_thinking:
            assistant_msg['thinking'] = full_thinking
        if tool_calls_reqs:
            assistant_msg['tool_calls'] = tool_calls_reqs
        llm_msgs.append(assistant_msg)
        
        if len(tool_calls_reqs) > 0:
            for tc in tool_calls_reqs:
                t_name = tc.function.name
                t_args = tc.function.arguments
                ptk.print_formatted_text(f"## Executing tool {t_name} with args {t_args}")

                t_result = None
                # TODO: Reduce code duplication related to subprocess running
                if t_name == 'bc':
                    try:
                        async with asyncio.timeout(20):
                            bc_proc = await asyncio.create_subprocess_exec(
                                'podman', 'run',
                                '-i', '--rm', 'ubuntu-with-stuff:latest',
                                'bc',
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE)
                            bc_expr = t_args['expr']
                            # ptk.print_formatted_text(f"{bc_expr=}, {type(bc_expr)=}")
                            (bc_stdout, bc_stderr) = await bc_proc.communicate(bytes(bc_expr + '\n', 'ascii'))
                            bc_stdout = bc_stdout.decode('ascii').strip()
                            bc_stderr = bc_stderr.decode('ascii').strip()
                            # ptk.print_formatted_text(f"{(bc_stdout, bc_stderr)=}")
                            # t_result = {'stdout': bc_stdout, 'stderr': bc_stderr}
                            t_result = bc_stdout
                    except TimeoutError:
                        t_result = 'Timeout'
                        ptk.print_formatted_text("Timeout in `bc`!")
                elif t_name == 'python_exec':
                    try:
                        async with asyncio.timeout(20):
                            py_proc = await asyncio.create_subprocess_exec(
                                'podman', 'run',
                                '-i', '--rm', 'ubuntu-with-stuff:latest',
                                'python3',
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE)
                            (py_stdout, py_stderr) = await py_proc.communicate(bytes(t_args['code'], 'utf-8'))
                            py_stdout = py_stdout.decode('utf-8').strip()
                            py_stderr = py_stderr.decode('utf-8').strip()
                            t_result = {'stdout': py_stdout, 'stderr': py_stderr}
                            # t_result = py_stdout
                    except TimeoutError:
                        t_result = 'Timeout'
                        ptk.print_formatted_text("Timeout in `python_exec`!")
                elif t_name == 'ed_editor':
                    try:
                        async with asyncio.timeout(20):
                            ed_proc = await asyncio.create_subprocess_exec(
                                'podman', 'run',
                                '-i', '--volume', '/tmp/cont:/root/ws',
                                'ubuntu-with-stuff:latest',
                                'ed', *([t_args['file']] if 'file' in t_args else []),
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE)
                            (ed_stdout, ed_stderr) = await ed_proc.communicate(bytes(t_args['instructions'], 'utf-8'))
                            ed_stdout = ed_stdout.decode('utf-8').strip()
                            ed_stderr = ed_stderr.decode('utf-8').strip()
                            t_result = ed_stdout
                    except TimeoutError:
                        t_result = 'Timeout'
                        ptk.print_formatted_text("Timeout in `ed_editor`!")
                else:
                    t_result = 'Unknown tool!'
                llm_msgs.append({'role': 'tool', 'tool_name': t_name, 'content': str(t_result)})
            new_user_msg = False

        if new_user_msg:
            input_evt.set()

if __name__ == '__main__':
    asyncio.run(main())
