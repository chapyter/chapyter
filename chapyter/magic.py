import argparse
import logging
import os
import re
from typing import Any

from IPython.display import display, Javascript
import tiktoken


import sys
from langchain.chat_models import ChatOpenAI
from .athena_utils import query_llm, sql_query_to_athena_df
from .programs import get_notebook_ordered_history

DATABASE = 'mimiciii'
S3_BUCKET = 's3://emmett-athena-bucket/'
AWS_REGION = 'us-east-1'
LLM = ChatOpenAI(model_name="gpt-3.5", max_tokens=2000)


import dotenv
import guidance
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import (  # type: ignore
    Magics,
    line_cell_magic,
    line_magic,
    magics_class,
    register_line_cell_magic

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
        **kwargs,
    ):
        
        #STEP 2: Executes chat
        program = self._get_program(args, chatonly=kwargs.pop("chatonly", False))
        llm = self._load_model(args, program)

        response = program.execute(
            message=message,
            llm=llm,
            shell=shell,
            sys_prompt=sys_prompt,
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
    def mimicSQL(self, line, cell=None):
        args = parse_argstring(self.mimicSQL, line)

        if cell is None:
            return
        current_message = cell

        overall_sys_prompt = """
                     You are a Medical AI Research Assistant, helping a Clinical Researcher do analyses of the MIMIC-III dataset on AWS Athena.

                     Use best guess ICD-9 codes, don't include decimals.
                     Instead of tables like 'mimic.mimiciii.patients' simply use 'patients'.
                     Never return more than one SQL query.
                     Dont use the following commands, because they won't work on AWS Athena: GROUP_CONCAT, string_agg, AGE, date_part
                     For date calculation, use Athena command DATE_DIFF.
                     When relevant, guess itemids rather than searching for them!
                     

                     When crafting the SQL query, think step-by-step - don't retrieve columns that are not present in the table!
                                          
                     If there is no dataset obvious to retrieve from, answer in general from your information and the past conversation.  

                     Return all SQL queries between two sets of triple ticks.                   
                     """
                
        context = get_notebook_ordered_history(current_message, os.getenv('NOTEBOOK_NAME'))

        program_out = self.execute_chat(context, args, self.shell, overall_sys_prompt)

        #regex to get rid of SQL in the response
        program_out_noSQL_list = program_out.split("```")
        program_out_noSQL_list = [s for s in program_out_noSQL_list if not s.startswith("sql")]
        
        #this contains all the non-SQL text
        program_out_noSQL = "".join(program_out_noSQL_list)
        program_out_noSQL = re.sub(r'\n+', '\n\n', program_out_noSQL)
        print(program_out_noSQL) #actually print it so that its in the history

        #take the element in the list that has SQL code
        #also remove the 'sql' that it starts with
        sql_query = [s for s in program_out.split("```") if s.startswith("sql")]
        sql_query = [s[3:] if s.startswith("sql") else s for s in sql_query]



        if len(sql_query) > 0:
            self.shell.set_next_input(f'%%runSQL \n\n{sql_query[0]}')



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
    def mimicRevealChatHistory(self, line, cell=None):
        #This line-magic allows you to see what is being fed to the LLM
        #Note, this requires only one '%', and not '%%'
        if cell is None:
            return
        current_message = cell

        context = get_notebook_ordered_history(current_message, os.getenv('NOTEBOOK_NAME'))
        print(context)

        print("\n", "-"*50, "\n")
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(context))
        print(f"Current history is {num_tokens} tokens!")



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

        args = parse_argstring(self.mimicPython, line)

        overall_sys_prompt = """
                     You are a Medical AI Research Assistant, helping a Clinical Researcher do analysis on their dataframe.

                     If possible, respond to the Clinical Researcher with python code to help them perform their analysis. Assume the past dataset is in a dataframe called 'df'.
                     !Don't assume anything about the datatypes in the df, because most of them are likely strings. Convert those strings as needed!
                     Any python code should end in a print statement showing the result.
                     If the conversation doesn't require python code at this moment, simply respond based on the past conversation and your knowledge of the dataset.     

                     Include ALL modules used.

                     All python code should be returned between the characters "```". In other words, your response should contain the following: 
                     ```
                     INSERT PYTHON CODE HERE
                     ```                        
                     """

        if cell is None:
            return
        current_message = cell

        context = get_notebook_ordered_history(current_message, os.getenv('NOTEBOOK_NAME'))

        program_out = self.execute_chat(context, args, self.shell, overall_sys_prompt)

        # First, check for code between "```"
        python_code = [s for s in program_out.split("```") if s and not s.startswith("python")]

        # If python_code's length is zero, use query_llm method
        if len(python_code) == 0:
            sys_prompt = """
            You are an expert coder. 
            If the text/code sent to you contains python code, respond with only the python code found. 
            Don't assume the datatypes in the table are anything. Assume in any df, that all values are strings and convert them to numbers as necessary.
            No comments. Should be directly executable. Assume we already have the dataframe called 'df', if one is needed. 
            MUST return a variable called 'answer' Otherwise respond only 'NO PYTHON CODE FOUND'.
            """
            result_from_llm = query_llm(program_out, sys_prompt, model_name="gpt-3.5-turbo")
            if result_from_llm != "NO PYTHON CODE FOUND":
                python_code = [result_from_llm]

        # Subtract the python_code from program_out to get program_out_noPython
        program_out_noPython = program_out
        for code in python_code:
            program_out_noPython = program_out_noPython.replace(code, "")

        # Continue with the rest of the function...

        # Now subtract the python_code from program_out to get program_out_noPython
        program_out_noPython = program_out
        for code in python_code:
            # Try subtracting the code as it is from program_out_noPython
            if code in program_out_noPython:
                program_out_noPython = program_out_noPython.replace(code, "")
            # If that doesn't work, try subtracting it with the "```" wrapping
            else:
                program_out_noPython = program_out_noPython.replace(f"```{code}```", "")

        program_out_noPython = re.sub(r'\n+', '\n\n', program_out_noPython)
        print(program_out_noPython)


        if len(python_code) > 0:
            self.shell.set_next_input(f'##AI-generated-code\n\n{python_code[0]}')

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
    def runSQL(self, line, cell=None):
        args = parse_argstring(self.runSQL, line)

        if cell is None:
            return
        current_message = cell

        #retrieve the df, and put it in notebook memory
        df, _ = sql_query_to_athena_df(current_message)
        display(df.head(5))
        self.shell.user_ns['df'] = df

        #add only the first two rows to llm_responses
        first_two_rows_str = df.head(2).to_string()


def load_ipython_extension(ipython):
    """
    Any module file that define a function named `load_ipython_extension`
    can be loaded via `%load_ext module.path` or be configured to be
    autoloaded by IPython at startup time.
    """
    ipython.register_magics(Chapyter)