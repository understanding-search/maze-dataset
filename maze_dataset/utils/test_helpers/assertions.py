import torch
from zanj.torchutil import ConfigMismatchException, assert_model_cfg_equality

from maze_dataset.training.config import ZanjHookedTransformer


def assert_model_output_equality(
    model_a: ZanjHookedTransformer, model_b: ZanjHookedTransformer
):
    try:
        assert_model_cfg_equality(model_a, model_b)
    except ConfigMismatchException as e:
        if e.diff == {
            "model_cfg": {"are_weights_processed": {"self": False, "other": True}}
        } or e.diff == {
            "model_cfg": {
                "are_layernorms_folded": {"self": False, "other": True},
                "are_weights_processed": {"self": False, "other": True},
            }
        }:
            pass
        else:
            raise e

    # Random input tokens
    dataset_cfg = model_a.zanj_model_config.dataset_cfg
    input_sequence = torch.randint(
        low=0,
        high=len(dataset_cfg.token_arr),
        size=(1, min(dataset_cfg.seq_len_max, 10)),
    )

    # (copied from `test_eval_model.py`)
    # Check for equality in argsort (absolute values won't be equal due to centering the unembedding weight matrix)
    assert torch.all(
        model_a(input_sequence.clone()).argsort()
        == model_b(input_sequence.clone()).argsort()
    )
    # apply normalization (e.g. softmax) and check with atol v-small
    # (roughly 1E-7 for float error on logexp I think)
    output_a = torch.nn.functional.softmax(model_a(input_sequence.clone()), dim=-1)
    output_b = torch.nn.functional.softmax(model_b(input_sequence.clone()), dim=-1)

    assert torch.allclose(output_a, output_b, atol=1e-7)
