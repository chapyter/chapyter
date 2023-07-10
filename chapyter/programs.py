import dataclasses
import re
from typing import Any, Callable, Dict, Optional

import guidance
from IPython.core.interactiveshell import InteractiveShell

__all__ = [
    "ChapyterAgentProgram",
    "_DEFAULT_PROGRAM",
    "_DEFAULT_HISTORY_PROGRAM",
]


@dataclasses.dataclass
class ChapyterAgentProgram:
    guidance_program: guidance.Program
    pre_call_hooks: Optional[Dict[str, Callable]]
    post_call_hooks: Optional[Dict[str, Callable]]
    model_name: Optional[str] = None

    def __post_init__(self):
        self.pre_call_hooks = self.pre_call_hooks or {}
        self.post_call_hooks = self.post_call_hooks or {}

    def execute(self, message: str, llm: str, shell: InteractiveShell, **kwargs) -> str:
        model_input_message: Any = message
        for name, hook in self.pre_call_hooks.items():
            model_input_message = hook(
                model_input_message,
                shell,
                **kwargs,
            )

        raw_program_response = self.guidance_program(
            **model_input_message, llm=llm, **kwargs
        )
        response = raw_program_response

        for name, hook in self.post_call_hooks.items():
            response = hook(
                response,
                shell,
                **kwargs,
            )
        return response


default_coding_guidance_program = guidance(
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


MARKDOWN_CODE_PATTERN = re.compile(r"`{3}([\w]*)\n([\S\s]+?)\n`{3}")


def clean_response_str(raw_response_str: str, shell: InteractiveShell, **kwargs):
    all_code_spans = []
    for match in MARKDOWN_CODE_PATTERN.finditer(raw_response_str):
        all_code_spans.append(match.span(2))

    # TODO: This is a very bold move -- if there is no code inside
    # markdown code block, we will assume that the whole block is code.
    # We need better ways to handle this in the future, e.g., checking
    # whether the first line of the output is valid python code.
    if len(all_code_spans) == 0:
        all_code_spans.append((0, len(raw_response_str)))

    cur_pos = 0
    all_converted_str = []
    for cur_start, cur_end in all_code_spans:
        non_code_str = raw_response_str[cur_pos:cur_start]
        non_code_str = "\n".join(
            [
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


_DEFAULT_PROGRAM = ChapyterAgentProgram(
    guidance_program=default_coding_guidance_program,
    pre_call_hooks={
        "wrap_to_dict": (lambda x, shell, **kwargs: {"current_message": x})
    },
    post_call_hooks={
        "extract_markdown_code": (
            lambda raw_response_str, shell, **kwargs: clean_response_str(
                raw_response_str["code"], shell, **kwargs
            )
        )
    },
)

python_interpreter_guidance_program = guidance(
    """
{{#system~}}
You are a helpful assistant to help with an python programmer.
{{~/system}}

{{#user~}}
Here is my python coding environment:
{{code_history}}
>>> %%chat
... {{current_prompt}}
Complete the code in a format like the one above.
{{~/user}}

{{#assistant~}}
{{gen 'code' temperature=0 max_tokens=2048}}
{{~/assistant}}
"""
)


def get_execution_history(
    ipython, print_nums=False, pyprompts=True, get_output=True, width=4
):
    def _format_lineno(session, line):
        """Helper function to format line numbers properly."""
        if session in (0, ipython.history_manager.session_number):
            return str(line)
        return "%s/%s" % (session, line)

    def limit_output(output, limit=100):
        """Limit the output to a certain number of words."""
        words = output.split()
        if len(words) > limit:
            return " ".join(words[:limit]) + "..."
        return output

    hist = ipython.history_manager.get_range_by_str(
        " ".join([]), True, output=get_output
    )
    # hist = ipython.history_manager.search(r"*", raw=True, output=get_output, n=100, unique=True)

    history_strs = []
    for session, lineno, inline in hist:
        history_str = ""
        inline, output = inline

        if inline.startswith("%load_ext"):
            continue
        inline = inline.expandtabs(4).rstrip()

        # Remove the assistant code template
        pattern = r"# Assistant Code for Cell \[\d+\]:"
        inline = re.sub(pattern, "", inline).strip()

        multiline = "\n" in inline
        line_sep = "\n" if multiline else " "

        if print_nums:
            history_str = history_str + "%s:%s" % (
                _format_lineno(session, lineno).rjust(width),
                line_sep,
            )

        if pyprompts:
            history_str = history_str + ">>> "
            if multiline:
                inline = "\n... ".join(inline.splitlines()) + "\n..."

        history_str = history_str + inline

        if get_output and output:
            history_str += "\n" + limit_output(output.strip())

        history_strs.append(history_str)
    return "\n".join(history_strs)


def clean_response_str_in_interpreter(raw_response_str):
    raw_response_str = raw_response_str.strip()
    # Remove the leading ">>>"
    raw_response_str = raw_response_str.lstrip(">>> ")
    # Split the string into lines
    lines = raw_response_str.split("\n... ")
    # Join the lines back together with newline characters
    raw_response_str = "\n".join(lines)
    # If the string ends with "...", remove it
    if raw_response_str.endswith("..."):
        raw_response_str = raw_response_str[:-3]

    return raw_response_str.strip()


_DEFAULT_HISTORY_PROGRAM = ChapyterAgentProgram(
    guidance_program=python_interpreter_guidance_program,
    pre_call_hooks={
        "add_execution_history": (
            lambda raw_message, shell, **kwargs: {
                "code_history": get_execution_history(shell),
                "current_prompt": raw_message,
            }
        )
    },
    post_call_hooks={
        "clean_response_str": (
            lambda raw_response_str, shell, **kwargs: clean_response_str_in_interpreter(
                raw_response_str["code"]
            )
        )
    },
)
