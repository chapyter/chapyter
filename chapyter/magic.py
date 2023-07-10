import argparse
import dataclasses
import logging
import os
from typing import Any, Optional, Union

import dotenv
import guidance
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import (  # type: ignore
    Magics,
    line_cell_magic,
    line_magic,
    magics_class,
)
from IPython.core.magic_arguments import (  # type: ignore
    argument,
    magic_arguments,
    parse_argstring,
)

from .programs import (
    _DEFAULT_CHATONLY_PROGRAM,
    _DEFAULT_HISTORY_PROGRAM,
    _DEFAULT_PROGRAM,
    ChapyterAgentProgram,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROGRAM_NAME = "_default"
_DEFAULT_HISTORY_PROGRAM_NAME = "_default_history"
_DEFAULT_CHATONLY_PROGRAM_NAME = "_default_chatonly"


@dataclasses.dataclass
class ChapyterAgentConfig:
    default_model: str = "gpt-4"
    assistance_code_hide: str = True

    # Program Configs
    default_program: str = _DEFAULT_PROGRAM_NAME
    default_history_program: str = _DEFAULT_HISTORY_PROGRAM_NAME
    default_chatonly_program: str = _DEFAULT_CHATONLY_PROGRAM_NAME


class ChapyterAgent:
    def __init__(self, config: ChapyterAgentConfig = None):
        self.config = config or ChapyterAgentConfig()
        self._programs = {}

        # Initialize default programs
        for program_name, program in [
            (_DEFAULT_PROGRAM_NAME, _DEFAULT_PROGRAM),
            (_DEFAULT_HISTORY_PROGRAM_NAME, _DEFAULT_HISTORY_PROGRAM),
            (_DEFAULT_CHATONLY_PROGRAM_NAME, _DEFAULT_CHATONLY_PROGRAM),
        ]:
            self.register_program(program_name, program)

    def _load_model(self, model_name: Optional[str]) -> guidance.llms.LLM:
        model = guidance.llms.OpenAI(
            model_name, organization=os.environ["OPENAI_ORGANIZATION"]
        )
        # TODO: Support Azure models
        return model

    def load_model(self, args, program: ChapyterAgentProgram) -> guidance.llms.LLM:
        model_name = args.model
        if model_name is None:
            model_name = self.config.default_model
        if program.model_name is not None:
            model_name = program.model_name

        return self._load_model(model_name)

    def _get_program(self, program_name: str = None) -> ChapyterAgentProgram:
        if program_name not in self._programs:
            raise ValueError(f"Program {program_name} not found.")
        return self._programs[program_name]

    def get_program(self, args, chatonly: bool = False) -> ChapyterAgentProgram:
        if args.program is None:
            if chatonly:
                return self._get_program(self.config.default_chatonly_program)
            if not args.history:
                return self._get_program(self.config.default_program)
            else:
                return self._get_program(self.config.default_history_program)
        else:
            if isinstance(args.program, guidance.Program):
                return ChapyterAgentProgram(args.program)
            else:
                try:
                    return self._get_program(args.program)
                except ValueError:
                    return ChapyterAgentProgram(guidance(args.program))

    def register_program(
        self,
        program_name: str,
        program: ChapyterAgentProgram,
    ):
        self._programs[program_name] = program
        logger.info(f"Registered template {program_name}.")

    def execute_chat(
        self,
        message: str,
        args: argparse.Namespace,
        shell: InteractiveShell,
        **kwargs,
    ):
        program = self.get_program(args, kwargs.get("chatonly", False))
        llm = self.load_model(args, program)

        response = program.execute(
            message=message,
            llm=llm,
            shell=shell,
            silent=not args.verbose,
            **kwargs,
        )
        return response


@magics_class
class Chapyter(Magics):
    def __init__(
        self,
        shell: InteractiveShell = None,
        agent: ChapyterAgent = None,
    ):
        super().__init__(shell)

        if agent is None:
            self.agent: ChapyterAgent = ChapyterAgent()

        if os.path.exists(".env"):
            dotenv.load_dotenv(".env")
            logger.info(f"Loaded .env file in the current directory ({os.getcwd()}).")

    @magic_arguments()
    @argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="The model to be used for the chat interface.",
    )
    @argument(
        "--history",
        "-h",
        action="store_true",
        help="Whether to use history for the code.",
    )
    @argument(
        "--program",
        "-p",
        type=Any,
        default=None,
        help="The program to be used for the chat interface.",
    )
    @argument(
        "--safe",
        "-s",
        action="store_true",
        help="Activate safe Mode that the code won't be automatically executed.",
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
            return
        current_message = cell

        program_out = self.agent.execute_chat(current_message, args, self.shell)
        execution_id = self.shell.execution_count
        program_out = f"# Assistant Code for Cell [{execution_id}]:\n" + program_out
        self.shell.run_cell(
            f"""get_ipython().set_next_input(\"\"\"{program_out}\"\"\")"""
        )

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
            return
        current_message = cell

        program_out = self.agent.execute_chat(
            current_message, args, self.shell, chatonly=True
        )
        print(program_out)

    @line_magic
    def chapyter_load_agent(self, line=None):
        """Reload the chapyter agent with all the configurations"""
        pass

    @line_cell_magic
    def chapyter_config(self, line, cell=None):
        """Configure the chapyter agent"""
        pass


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    # You can register the class itself without instantiating it.  IPython will
    # call the default constructor on it.
    ipython.register_magics(Chapyter)
