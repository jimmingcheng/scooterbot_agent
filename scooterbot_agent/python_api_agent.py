from abc import ABC
from abc import abstractmethod
import datetime
import inspect
import json
import textwrap
from openai import OpenAI
from tenacity import retry
from tenacity import stop_after_attempt
from typing import get_type_hints


from .private_agent import PrivateAgent


class PythonAPIAgent(PrivateAgent, ABC):
    """Base class for agents that interact with python APIs."""

    def reply(self, message: str) -> str:
        return self.answer_with_api(message)

    @abstractmethod
    def overview(cls) -> str:
        """Succinct description of what this API does. To be read by LLM to decide whether to use
        this API."""

    @abstractmethod
    def usage_guide(self) -> str:
        """Detailed description of the API's capabilities and how to invoke them. Should instruct
        the LLM to call the `invoke_api` tool"""

    @abstractmethod
    def invoke_api(self, **args) -> str:
        """Implements the `invoke_api` tool."""

    @abstractmethod
    def tool_spec_for_invoke_api(self) -> dict:
        """OpenAI tool spec for the `invoke_api` tool."""

    @classmethod
    def format_prior_state(cls, state: str) -> str:
        return textwrap.dedent(
            """\
            ====== STATE AS OF {datetime} ======
            {state}
            """
        ).format(
            datetime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            state=state,
        )

    def initial_state(self) -> str:
        return ''

    @retry(stop=stop_after_attempt(3))
    def answer_with_api(
        self,
        request: str,
        prior_states: str | None = None,
        depth: int = 0,
        max_depth: int = 3
    ) -> str:
        if not prior_states:
            prior_states = self.format_prior_state(self.initial_state())

        system_prompt = textwrap.dedent(
            '''\
            # Instructions

            You are given `request` and `prior_states`.

            1. As soon as `prior_states` is sufficient to answer request, create and return `answer` (plain English)
            2. Else, call `invoke_api` to fetch more data, so the process can be repeated with the additional data

            # API Usage Guide

            {usage_guide}
            '''
        ).format(
            usage_guide=self.usage_guide()
        )

        user_prompt = textwrap.dedent(
            '''\
            # Request

            {request}

            # Prior States

            {prior_states}
            '''
        ).format(
            request=request,
            prior_states=prior_states,
        )

        completion_message = OpenAI().chat.completions.create(
            messages=[
                {'role': 'user', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            model='gpt-4o',
            tools=[
                self.tool_spec_for_invoke_api(),  # type: ignore
                {
                    "type": "function",
                    "function": {
                        "name": "answer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                            },
                            "required": ["answer"],
                        }
                    }
                }
            ],
            tool_choice='required',
        ).choices[0].message

        if completion_message.tool_calls:
            assert completion_message.tool_calls

            tool_call = completion_message.tool_calls[0]

            func_args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == 'invoke_api':
                new_state = self.invoke_api(**func_args)
                prior_states += '\n\n' + self.format_prior_state(new_state)
                if depth < max_depth:
                    return self.answer_with_api(request, prior_states, depth + 1, max_depth)
                else:
                    return "Sorry, I ran out of steps"
            elif tool_call.function.name == 'answer':
                return func_args['answer']
            raise ValueError('Unexpected tool call')
        else:
            assert completion_message.content
            return completion_message.content


def generate_python_api_doc(cls: type, whitelisted_members: list[str] | None = None) -> str:
    lines = [f'class {cls.__name__}:']

    class_doc = inspect.getdoc(cls)
    if class_doc:
        lines.append(f'    """{class_doc}"""\n')

    annotations = get_type_hints(cls, localns=cls.__dict__)  # type: ignore

    def get_type_name(annotation):
        """Helper function to extract only the class name from a type hint."""
        if annotation is None or annotation is inspect.Signature.empty:
            return "None"
        if isinstance(annotation, type):
            return annotation.__name__
        if hasattr(annotation, "__origin__"):  # Handles generics like `list[Vehicle]`
            origin = annotation.__origin__.__name__
            args = ", ".join(get_type_name(arg) for arg in annotation.__args__)
            return f"{origin}[{args}]"
        return str(annotation)

    if not whitelisted_members:
        whitelisted_members = list(annotations.keys()) + [
            name for name, _ in inspect.getmembers(cls) if not name.startswith("_")
        ]

    for name in whitelisted_members:
        if hasattr(cls, name):
            attr = getattr(cls, name)
            if inspect.isfunction(attr) or inspect.ismethod(attr):  # If it's a function/method
                sig = inspect.signature(attr)
                params = [
                    f"{param.name}: {get_type_name(param.annotation)}"
                    if param.annotation is not inspect.Parameter.empty
                    else param.name
                    for param in sig.parameters.values()
                ]
                return_type = f" -> {get_type_name(sig.return_annotation)}"
                line = f'    def {name}({", ".join(params)}){return_type}'
            else:  # If it's a variable
                line = f'    {name}: {type(attr).__name__}'
            lines.append(line)
        elif name in annotations:
            lines.append(f'    {name}: {get_type_name(annotations[name])}')

    return "\n".join(lines)
