import json
import os
import re

import guidance
from IPython.core.magic import (  # type: ignore
    Magics,
    cell_magic,
    line_cell_magic,
    line_magic,
    magics_class,
)
from IPython.core.magic_arguments import (  # type: ignore
    argument,
    magic_arguments,
    parse_argstring,
)


def load_model(model_name: str):
    model = guidance.llms.OpenAI(
        model_name, organization=os.environ["OPENAI_ORGANIZATION"]
    )
    return model


MARKDOWN_CODE_PATTERN = re.compile(r"`{3}([\w]*)\n([\S\s]+?)\n`{3}")
DEFAULT_PROGRAM = guidance(
    """
{{#system~}}
You are a helpful and assistant and you are chatting with an python programmer.

{{~/system}}

{{#user~}}
From now on, you are ought to generate only the python code based on the description from the programmer.
{{~/user}}

{{#assistant~}}
Ok, I will do that. Let's do a practice round.
{{~/assistant}}

{{#user~}}
Load the json file called orca.json
{{~/user}}

{{#assistant~}}
import json 
with open('orca.json') as file:
    data = json.load(file)
{{~/assistant}}

{{#user~}}
That was great, now let's do another one.
{{~/user}}

{{#assistant~}}
Sounds good.
{{~/assistant}}

{{#user~}}
{{current_message}}
{{~/user}}

{{#assistant~}}
{{gen 'code' temperature=0 max_tokens=2048}}
{{~/assistant}}
"""
)


def clean_response_str(raw_response_str):
    all_code_spans = []
    for match in MARKDOWN_CODE_PATTERN.finditer(raw_response_str):
        all_code_spans.append(match.span(2))
    cur_pos = 0
    all_converted_str = []
    for cur_start, cur_end in all_code_spans:
        non_code_str = raw_response_str[cur_pos:cur_start]
        non_code_str = "\n".join(
            ["# Assistant:"]
            + [
                f"# {ele}"
                for ele in non_code_str.split("\n")
                if not ele.startswith("```") and ele.strip()
            ]
        )
        code_str = raw_response_str[cur_start:cur_end].strip()
        cur_pos = cur_end
        all_converted_str.extend([non_code_str, code_str])

    last_non_code_str = [
        f"#{ele}"
        for ele in raw_response_str[cur_pos:].split("\n")
        if not ele.startswith("```") and ele.strip()
    ]
    if len(last_non_code_str) > 0:
        all_converted_str.append("\n".join(last_non_code_str))

    return "\n".join(all_converted_str)


# The class MUST call this class decorator at creation time
@magics_class
class Chapyter(Magics):
    @line_magic
    def chat(self, line):
        "my line magic"
        print("Full access to the main IPython object:", self.shell)
        print("Variables in the user namespace:", list(self.shell.user_ns.keys()))
        return line

    @magic_arguments()
    @argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4",
        help="The model to be used for the chat interface.",
    )
    @argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to set slient=True for guidance calls.",
    )
    @line_cell_magic
    def chat(self, line, cell=None):
        args = parse_argstring(self.chat, line)

        if cell is None:
            # print("Called as line magic")
            current_message = line
        else:
            # print("Called as cell magic")
            current_message = cell

        model = load_model(args.model)
        program_out = DEFAULT_PROGRAM(
            current_message=current_message,
            llm=model,
            silent=not args.verbose,
        )
        # print(program_out["code"])
        # self.shell.run_cell(program_out["code"])
        # code_span = MARKDOWN_CODE_PATTERN.findall(program_out["code"])
        code_str = clean_response_str(program_out["code"])
        self.shell.run_cell(f"""get_ipython().set_next_input(\"\"\"{code_str}\"\"\")""")

    @magic_arguments()
    @argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4",
        help="The model to be used for the chat interface.",
    )
    @argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether to set slient=True for guidance calls.",
    )
    @line_cell_magic
    def chatonly(self, line, cell=None):
        args = parse_argstring(self.chat, line)

        if cell is None:
            # print("Called as line magic")
            current_message = line
        else:
            # print("Called as cell magic")
            current_message = cell

        model = load_model(args.model)
        program_out = DEFAULT_PROGRAM(
            current_message=current_message,
            llm=model,
            silent=not args.verbose,
        )
        return program_out["code"]


# In order to actually use these magics, you must register them with a
# running IPython.


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(Chapyter)
