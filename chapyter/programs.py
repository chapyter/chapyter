#Step 1: %mimic cell magic triggered
#Step 2: def execute_chat (doesn't do anything interesting, passes on message)
#Step 3: def execute (doesn't do anything interesting, passes on message)
#Step 4: guidance_program - this is where we input our custom phrasing

import dataclasses
import re
from typing import Any, Callable, Dict, Optional
import openai
import nbformat
import os



import guidance
from IPython.core.interactiveshell import InteractiveShell

__all__ = [
    "ChapyterAgentProgram",
    "_DEFAULT_HISTORY_PROGRAM",
]


def extract_code_blocks(text):
    # Regular expression pattern to match code blocks between triple backticks
    pattern = r"```(.*?)```"
    
    # Use re.DOTALL to make sure '.' matches newline characters as well
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Strip any leading or trailing whitespace from each match and return as a list
    return [match.strip() for match in matches]


def query_llm(llm_prompt, sys_prompt):
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": llm_prompt}
        ],
        max_tokens=1000,
        temperature=0.1,
    )
    response = response["choices"][0]["message"]["content"]
    return response


@dataclasses.dataclass
class ChapyterAgentProgram:
    guidance_program: guidance.Program
    pre_call_hooks: Optional[Dict[str, Callable]]
    post_call_hooks: Optional[Dict[str, Callable]]
    model_name: Optional[str] = None

    def __post_init__(self):
        self.pre_call_hooks = self.pre_call_hooks or {}
        self.post_call_hooks = self.post_call_hooks or {}

    #This is step 3, execute
    def execute(self, message: str, llm: str, shell: InteractiveShell, sys_prompt: str, llm_responses: list, **kwargs) -> str:

        llm_response = query_llm(message, sys_prompt)

        # print("\n\n")
        # print(llm_prompt, "\n\n")
        # print("AI response: ")
        # print(llm_response)

        return llm_response


MARKDOWN_CODE_PATTERN = re.compile(r"`{3}([\w]*)\n([\S\s]+?)\n`{3}")


def clean_response_str(raw_response_str: str):
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
        f"# {ele}"
        for ele in raw_response_str[cur_pos:].split("\n")
        if not ele.startswith("```") and ele.strip()
    ]
    if len(last_non_code_str) > 0:
        all_converted_str.append("\n".join(last_non_code_str))

    return "\n".join(all_converted_str)


def clean_execution_history(s):
    # Remove triple backticks
    s = s.replace('```', '')
    
    # Remove leading and closing newline characters
    s = s.strip()
    
    # Remove the %%mimic --safe -h
    s = s.replace('%%mimicSQL', '').strip()
    s = s.replace('%%mimicPython', '').strip()

    
    return s


import nbformat

import pprint

import pandas as pd

from IPython.core.display import display, HTML


def extract_table(outputs):
    for output in outputs:
        if 'data' in output and 'text/plain' in output['data']:
            return output['data']['text/plain']
    return None

def extract_text(outputs):
    for output in outputs:
        if 'text' in output:
            return output['text']
    return None


def get_notebook_ordered_history(current_message, notebook_name=None):

    #Extract "mimic" Human cells, keep them in order
    #Extract remaining AI cells, order doesnt matter

    #For each Human cell:
    #(1) Append Human input
    #(2) Identify relevant AI cell, append AI code response
    #(3)Then append Human output

    # Load the current notebook
    notebook_name = os.getenv("NOTEBOOK_NAME")

    with open(notebook_name, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    top_to_bottom_human_cells_inputs = []
    top_to_bottom_human_cells_output_tables = []
    top_to_bottom_human_cells_output_text = []

    # print("CELLS", nb.cells)

    for cell in nb.cells:

        # print("\n\n", cell)

        cell_input = cell["source"]
        cell_execution_count = cell["execution_count"]
        if "%%mimic" in cell_input:

            #if in a mimic cell, take the input
            top_to_bottom_human_cells_inputs.append(cell_input.replace("\n\n", " --- "))

            #break if this is the current cell
            # print(f"Comparing '{current_message.strip()}' to '{cell_input.strip()}'")
            if current_message.strip() in cell_input.strip():
                # print("Breaking!!!")
                break

            # top_to_bottom_execution_counts.append(cell_execution_count) 

            #if a table is in the outputs, grab it!!!
            if "outputs" in cell:
                outputs = cell["outputs"]

                table = extract_table(outputs)
                text = extract_text(outputs)

                top_to_bottom_human_cells_output_tables.append(table)
                top_to_bottom_human_cells_output_text.append(text)
                    

        else:
            if "Assistant Code" in cell_input:
                parts = cell_input.split("]:\n")
                execution_response_no = parts[0].split("[")[-1]
                code_generated = parts[1]
                # raw_llm_output_dict[int(execution_response_no)] = code_generated


    # print("\n\nGot outputs", top_to_bottom_human_cells_outputs)
    # print("LENGTHS", len(top_to_bottom_human_cells_inputs), len(top_to_bottom_human_cells_output_text), len(top_to_bottom_human_cells_output_tables))
    context = "="*60
    context += "\n"
    for human_input, AI_text, AI_table in zip(top_to_bottom_human_cells_inputs, top_to_bottom_human_cells_output_text, top_to_bottom_human_cells_output_tables):
        # print("In loop")
        context += f"**Clinical Researcher:** {human_input}\n\n"
        if AI_table != None:
            context += f"**AI Research Assistant:** {AI_text}\n{AI_table}\n\n"
        else:
            context += f"**AI Research Assistant:** {AI_text}\n\n"
        context += "="*60
        context += "\n\n"

    # print(top_to_bottom_human_cells_inputs)
    # print(top_to_bottom_human_cells_outputs)
    # print([raw_llm_output_dict[i] for i in top_to_bottom_execution_counts])

    context += f"**Clinical Researcher:** {top_to_bottom_human_cells_inputs[-1]}\n\n"
    context += f"**AI Research Assistant:**"




    from IPython.display import display, Markdown

    # print("Seeing pre-prompt")
    # display(Markdown(context))
    # print(context)

    return context

def get_execution_history(ipython, get_output=True, width=4):
    def limit_output(output, limit=100):
        """Limit the output to a certain number of words."""
        words = output.split()
        if len(words) > limit:
            return " ".join(words[:limit]) + "..."
        return output

    hist = ipython.history_manager.get_range_by_str(
        " ".join([]), True, output=get_output
    )

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

        if inline.startswith("%%chat"):
            inline = "\n".join(inline.splitlines()[1:])
            inline = inline.strip()
            history_str = history_str + inline
        else:
            history_str = history_str + "```\n" + inline + "\n```"

        if get_output and output:
            history_str += "\nOutput:\n" + limit_output(output.strip())

        history_strs.append(history_str + "\n")

    history_strs = [clean_execution_history(s) for s in history_strs]

    return history_strs


default_coding_history_guidance_program = guidance(
    """
{{#system~}}
You are a helpful and assistant and you are chatting with an programmer interested in retrieving data from the MIMIC-III SQL database on AWS Athena.
If they ask for something that is answerable with a SQL query, make sure there is only one SELECT statement.
{{~/system}}

{{#user~}}
Here is my code so far:
{{llm_conversation}}
{{~/user}}

{{#assistant~}}
{{gen 'code' temperature=0 max_tokens=2048}}
{{~/assistant}}
"""
)


_DEFAULT_HISTORY_PROGRAM = ChapyterAgentProgram(
    guidance_program=default_coding_history_guidance_program,
    pre_call_hooks={
        "add_execution_history": (
            lambda raw_message, shell, **kwargs: {
                "llm_conversation": get_execution_history(shell),
            }
        )
    },
    post_call_hooks={
        "extract_markdown_code": (
            lambda raw_response_str, shell, **kwargs: clean_response_str(
                raw_response_str["code"]
            )
        )
    },
)