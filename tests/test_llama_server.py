from deep_research.tools.llama_server import build_launch_command


def test_launch_command_includes_jinja_by_default():
    cmd = build_launch_command("/models/m.gguf", 18080)
    assert "--jinja" in cmd


def test_launch_command_jinja_can_be_disabled_per_model():
    """Escape hatch for a GGUF whose embedded chat template is broken or
    unsupported by llama.cpp's minja engine."""
    cmd = build_launch_command("/models/m.gguf", 18080, {"jinja": False})
    assert "--jinja" not in cmd


def test_launch_command_chat_template_file_override():
    cmd = build_launch_command("/models/m.gguf", 18080, {"chat_template_file": "/tmp/t.jinja"})
    assert cmd[cmd.index("--chat-template-file") + 1] == "/tmp/t.jinja"
