from IPython.terminal.interactiveshell import TerminalInteractiveShell

from chapyter.magic import Chapyter


def test_chapyter_config():
    shell = TerminalInteractiveShell()
    shell.run_line_magic("load_ext", "chapyter")

    # Test whether it can display the help message
    result = shell.run_line_magic("chapyter", "")

    # Test whether it can display the correct default values
    traits = Chapyter.class_own_traits()
    for key, value in traits.items():
        result = shell.run_line_magic("chapyter", key)

        # Skip for cases where the default value is None or empty string
        # TODO: check why there is such inconsistency
        if result is None and value.default_value == "":
            continue

        assert result == value.default_value

    # Test whether it can be used to properly set the values
    for key, value in traits.items():
        if isinstance(value, str):
            shell.run_line_magic("chapyter", f"{key}='test'")
            result = shell.run_line_magic("chapyter", key)
            assert result == "test"
        # Skip for other types for now

    # Test whether it can be used to properly set the values
    # For cases where there is a space between the value and the equal sign
    # See https://github.com/chapyter/chapyter/issues/36
    for key, value in traits.items():
        if isinstance(value, str):
            shell.run_line_magic("chapyter", f"{key} = 'text'")
            result = shell.run_line_magic("chapyter", key)
            assert result == "text"
