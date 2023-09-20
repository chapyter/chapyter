import argparse
import logging
import os
import re
from typing import Any, Optional, Union  # noqa

from IPython.display import display


import sys
from langchain.chat_models import ChatOpenAI
from .LLMimic_app import query_llm, sql_query_to_athena_df
from .programs import get_notebook_ordered_history

DATABASE = 'mimiciii'
S3_BUCKET = 's3://emmett-athena-bucket/'
AWS_REGION = 'us-east-1'
LLM = ChatOpenAI(model_name="gpt-4", max_tokens=2000)


import dotenv
import guidance
from IPython.core.error import UsageError
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
from traitlets import Bool, Dict, Instance, Unicode, default, observe  # noqa
from traitlets.config.loader import Config

from .programs import (
    _DEFAULT_HISTORY_PROGRAM,
    ChapyterAgentProgram,
)

llm_responses = []

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_PROGRAM_NAME = "_default_history"



@magics_class
class Chapyter(Magics):
    """
    The Chapyter Magic Command is used for handling the calls to the large language models.
    """

    # General Configs
    default_api_type = Unicode(
        "openai",
        help="""The default type of api that will be used to query the models. 
Currently we only support the following:
    - openai
    - azure
Will add more soon.
""",
    ).tag(config=True)

    # OpenAI API Configs
    openai_default_model = Unicode(
        "gpt-4",
        help=(
            "The default model that will be used to query the OpenAI API. "
            "This can be overridden by the `--model` flag."
        ),
    ).tag(config=True)

    openai_api_key = Unicode(
        allow_none=True,
        help=(
            "The API key used for OpenAI API queries. "
            "By default this will be read from the `OPENAI_API_KEY` environment variable. "  # noqa
            "Can be left empty if not using OpenAI."
        ),
    ).tag(config=True)

    @default("openai_api_key")
    def _default_openai_api_key(self):
        return os.environ.get("OPENAI_API_KEY", None)

    openai_api_org = Unicode(
        allow_none=True,
        help=(
            "The organization ID for OpenAI API. "
            "By default this will be read from the `OPENAI_ORGANIZATION` environment variable. "  # noqa
            "Can be left empty if not using OpenAI."
        ),
    ).tag(config=True)

    @default("openai_api_org")
    def _default_openai_api_org(self):
        return os.environ.get("OPENAI_ORGANIZATION", None)

    # Azure API Configs
    azure_openai_default_model = Unicode(
        allow_none=True,
        help=(
            "The default model used for Azure API queries. "
            "Can be overridden by the --model flag."  # TODO:
        ),
    ).tag(config=True)

    azure_openai_default_deployment_id = Unicode(
        allow_none=True,
        help=(
            "The default deployment id for Azure API. "
            "Different from OpenAI API, Azure API requires a deployment id to be specified. "
            "Can be left empty if not using Azure."
        ),
    ).tag(config=True)

    azure_openai_api_base = Unicode(
        allow_none=True,
        help="The base URL for Azure API. Can be left empty if not using Azure.",
    ).tag(config=True)

    azure_openai_api_version = Unicode(
        allow_none=True,
        help="The version of Azure API being used. Can be left empty if not using Azure.",
    ).tag(config=True)

    azure_openai_api_key = Unicode(
        allow_none=True,
        help=(
            "The API key used for Azure API queries. "
            "By default this will be read from the `AZURE_OPENAI_API_KEY` environment variable. "  # noqa
            "Can be left empty if not using Azure."
        ),
    ).tag(config=True)

    # Program Configs

    @default("azure_api_key")
    def _default_azure_api_key(self):
        return os.environ.get("AZURE_OPENAI_API_KEY", None)

    def __new__(cls, *args, **kwargs):
        # Load the .env file if it exists

        if os.path.exists(".env"):
            dotenv.load_dotenv(".env", override=True)
            logger.info(f"Loaded .env file in the current directory ({os.getcwd()}).")

        instance = super(Chapyter, cls).__new__(cls)
        return instance

    def __init__(
        self,
        shell: InteractiveShell = None,
    ):
        super().__init__(shell)

        # Initialize default programs
        self._programs = {}
        for program_name, program in [
            (_DEFAULT_HISTORY_PROGRAM_NAME, _DEFAULT_HISTORY_PROGRAM),
        ]:
            self._register_program(program_name, program)

    def _register_program(
        self,
        program_name: str,
        program: ChapyterAgentProgram,
    ):
        self._programs[program_name] = program
        logger.info(f"Registered template {program_name}.")

    def _load_model(
        self,
        args: argparse.Namespace,
        program: ChapyterAgentProgram,
    ) -> guidance.llms.LLM:
        """Load the appropriate model based on the arguments passed in.
        The resolution order is as follows:

        1. If the `--model` flag is passed in, use that model.
        2. Otherwise use the default model specified in the config.
        3. Otherwise use the default model specified in the program.
        """
        model_name = args.model

        if self.default_api_type == "openai":
            model_name = model_name or self.openai_default_model or program.model_name
            model = guidance.llms.OpenAI(
                model_name,
                api_type="openai",
                api_key=self.openai_api_key,
                organization=self.openai_api_org,
            )
            logger.info(f"Loaded model {model_name} from OpenAI API.")
        elif self.default_api_type == "azure":
            model_name = (
                model_name or self.azure_openai_default_model or program.model_name
            )
            model = guidance.llms.OpenAI(
                model_name,
                api_type="azure",
                api_key=self.azure_openai_api_key,
                api_base=self.azure_openai_api_base,
                api_version=self.azure_openai_api_version,
                deployment_id=self.azure_openai_default_deployment_id,
            )
            logger.info(
                f"Loaded model {model_name} ({self.azure_openai_default_deployment_id}) "
                "from Azure OpenAI API."
            )
        else:
            raise ValueError(
                f"Invalid api type {self.default_api_type}. "
                "Currently we only support the following: \n"
                "- openai \n"
                "- azure"
            )
        return model

    def _get_program(
        self,
        args: argparse.Namespace,
        chatonly: bool = False,
    ) -> ChapyterAgentProgram:
        if args.program is None:
 
            return self._programs[_DEFAULT_HISTORY_PROGRAM_NAME]
        else:
            # TODO: This part is a bit messy, need to clean up
            # So the current logic is that we allow users to pass in a program
            # either as a string or as a guidance.Program object.
            # If it's a guidance.Program object, we will use that directly.
            # If it's a string, we will try to load the program from the registry.
            # If it's not in the registry, we will try to load it directly into
            # a guidance.Program object.

            if isinstance(args.program, guidance.Program):
                return ChapyterAgentProgram(args.program)
            else:
                try:
                    return self._programs[args.program]
                except ValueError:
                    return ChapyterAgentProgram(guidance(args.program))

    def execute_chat(
        self,
        message: str,
        args: argparse.Namespace,
        shell: InteractiveShell,
        sys_prompt: str,
        llm_responses: list,
        **kwargs,
    ):
        
        #STEP 2: Executes chat
        program = self._get_program(args, chatonly=kwargs.pop("chatonly", False))
        llm = self._load_model(args, program)

        response = program.execute(
            message=message,
            llm=llm,
            shell=shell,
            silent=not args.verbose,
            sys_prompt=sys_prompt,
            llm_responses=llm_responses,
            **kwargs,
        )
        return response

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

        program_out = self.execute_chat(current_message, args, self.shell)
        execution_id = self.shell.execution_count

        #Emmett for executing SQl code
        sys_prompt = "You are an expert coder. If the text/code sent to you contains a SQL query, respond with only the SQL query found, which is directly executable in Athena. Don't return the query in the form of a triple string. Change things that look like 'mimic.mimiciii.patients' to simply 'patients'. Otherwise respond only 'NO SQL FOUND'."
        contains_SQL = query_llm(program_out, sys_prompt)
        if contains_SQL != "NO SQL FOUND": #If the response has SQL in it, execute it, get df and store it
            df, sql_query = sql_query_to_athena_df(contains_SQL)
            display(df.head(5))
            self.shell.user_ns['df'] = df

        program_out = f"# AAssistant Code for Cell [{execution_id}]:\n" + program_out
        # program_out = f"# Assistant Code for Cell [{execution_id}]:\n" + sql_query

        # self.shell.set_next_input(program_out)


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
    def mimicSQL(self, line, cell=None):
        args = parse_argstring(self.chat, line)

        if cell is None:
            return
        current_message = cell

        overall_sys_prompt = """
                     You are a Medical AI Research Assistant, helping a Clinical Researcher do analyses of the MIMIC-III dataset on AWS Athena.
                     If possible, respond to the Clinical Researcher with a SQL query to retrieve their relevant dataset.
                     If there is no dataset obvious to retrieve from, answer in general from your information and the past conversation.                     
                     """
                
        context = get_notebook_ordered_history(current_message)

        program_out = self.execute_chat(context, args, self.shell, overall_sys_prompt, llm_responses)
        print(program_out)

        execution_id = self.shell.execution_count

        #Emmett for executing SQl code
        sys_prompt = "You are an expert coder. If the text/code sent to you contains a SQL query, respond with only the SQL query found, which is directly executable in Athena. Don't return the query in the form of a triple string. Change things that look like 'mimic.mimiciii.patients' to simply 'patients'. Otherwise respond only 'NO SQL FOUND'."
        contains_SQL = query_llm(program_out, sys_prompt)
        if contains_SQL != "NO SQL FOUND": #If the response has SQL in it, execute it, get df and store it
            df, sql_query = sql_query_to_athena_df(contains_SQL)
            display(df.head(5))
            self.shell.user_ns['df'] = df

            first_two_rows_str = df.head(2).to_string()
            to_llm_history = program_out + "\n" + "First two rows of the df below:" + "\n" + first_two_rows_str        
            llm_responses.append(to_llm_history)
        else:
            llm_responses.append(program_out)



        program_out = f"# Assistant Code for Cell [{execution_id}]:\n"# + program_out
        # program_out = f"# Assistant Code for Cell [{execution_id}]:\n" + sql_query

        # self.shell.set_next_input(program_out)


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
    def mimicPython(self, line, cell=None):
        args = parse_argstring(self.chat, line)

        overall_sys_prompt = """
                     You are a Medical AI Research Assistant, helping a Clinical Researcher do analysis on their dataframe.
                     If possible, respond to the Clinical Researcher with python code to help them perform their analysis. Assume the past dataset is in a dataframe called 'df'.
                     If the conversation doesn't require python code at this moment, simply respond based on the past conversation and your knowledge of the dataset.                     
                     """



        if cell is None:
            return
        current_message = cell

        context = get_notebook_ordered_history(current_message)
        

        program_out = self.execute_chat(context, args, self.shell, overall_sys_prompt, llm_responses)

        print(program_out)

        execution_id = self.shell.execution_count

        sys_prompt = """You are an expert coder. 
        If the text/code sent to you contains python code, respond with only the python code found. If no python code found, respond only with 'NO PYTHON CODE FOUND'.
        Don't assume the datatypes in the table are anything. Assume in any df, that all values are strings and convert them to numbers as necessary.
        No comments. Should be directly executable. Assume we already have the dataframe called 'df', if one is needed. 
        If python code found, reformulate and return the python code so that it returns a variable called 'answer'.
        """

        contains_python = query_llm(program_out, sys_prompt)
        if contains_python != "NO PYTHON CODE FOUND": # and isinstance(st.session_state.df, pd.DataFrame):

            #Now that we got the code, lets analyze our df, need to retrieve from session state dict            
            #And then we will execute the code on the fly
            # print("Going to exec", contains_python)

            try:
                df = self.shell.user_ns['df']
                context = {'df': df}
                # print("SHAPAE OF THINGS", df.shape)
                # print(df['gender'].value_counts())
                exec(contains_python, globals(), context)
                answer = context.get('answer', None)

            except Exception as e:
                print("GOT EXCEPTION", e)
                exec(contains_python, globals())
                answer = context.get('answer')

            calculated_answer_string = f"Result : {answer}"
            print("\n", calculated_answer_string)
        else:
            # print(program_out)
            calculated_answer_string = ""

        to_llm_history = program_out + "\n" + calculated_answer_string
        llm_responses.append(to_llm_history)

        program_out = f"# Assistant Code for Cell [{execution_id}]:\n" #+ program_out + "\n" + calculated_answer_string + "\n"
        # self.shell.set_next_input(program_out)



    @line_magic
    def chapyter_load_agent(self, line=None):
        """Reload the chapyter agent with all the configurations"""
        pass

    @line_magic
    def chapyter(self, line):
        """Used for displaying and modifying Chapyter Agent configurations.

        Exemplar usage:

        - %chapyter
          print all the configurable parameters and its current value

        - %chapyter <parameter_name>
          print the current value of the parameter

        - %chapyter <parameter_name>=<value>
          set the value of the parameter
        """

        # This code is inspired by the %config magic command in IPython
        # See the code here: https://github.com/ipython/ipython/blob/6b17e43544316d691376e35e677032a4b00d6eeb/IPython/core/magics/config.py#L36

        # remove text after comments
        line = line.strip().split("#")[0].strip()

        all_class_configs = self.class_own_traits()

        if not line or line.startswith("#"):
            help = self.class_get_help(self)
            # strip leading '--' from cl-args:
            help = re.sub(re.compile(r"^--", re.MULTILINE), "", help)
            print(help)
            return
        elif line in all_class_configs.keys():
            return getattr(self, line)
        elif "=" in line and line.split("=")[0].strip() in all_class_configs.keys():
            cfg = Config()
            exec(f"cfg.{self.__class__.__name__}." + line, self.shell.user_ns, locals())
            self.update_config(cfg)
        elif line.startswith("help"):
            print(
                "The %chapyter magic command supports the following usage:\n"
                "- %chapyter\n  print all the configurable parameters and its current value\n"
                "- %chapyter <parameter_name>\n  print the current value of the parameter\n"
                "- %chapyter <parameter_name>=<value>\n  set the value of the parameter"
            )
        else:
            raise UsageError(
                f"Invalid usage of the chapyter command: {line}. "
                "It supports the following usage:\n"
                "- %chapyter\n  print all the configurable parameters and its current value\n"
                "- %chapyter <parameter_name>\n  print the current value of the parameter\n"
                "- %chapyter <parameter_name>=<value>\n  set the value of the parameter"
            )


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    ipython.register_magics(Chapyter)