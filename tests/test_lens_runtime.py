from swarmstop.lens_runtime import _resolve_text_stack


class Stack:
    layers = [object()] * 32
    norm = object()
    embed_tokens = object()


class Inner:
    language_model = Stack()


class Wrapper:
    model = Inner()
    lm_head = object()


def test_resolve_qwen_multimodal_text_stack():
    stack, layers, norm, head = _resolve_text_stack(Wrapper())

    assert stack is Wrapper.model.language_model
    assert len(layers) == 32
    assert norm is Wrapper.model.language_model.norm
    assert head is Wrapper.lm_head
