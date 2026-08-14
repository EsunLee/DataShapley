import torch

from gshap.model import PlovadDecoder


def test_model_shape_and_parameter_count():
    model = PlovadDecoder()
    assert model(torch.randn(7, 512)).shape == (7,)
    assert model.parameter_count == 65_793
    assert model.classifier[0].kernel_size == (1,)
    assert model.classifier[2].kernel_size == (1,)


def test_conv1d_matches_manual_linear_path():
    torch.manual_seed(1)
    model = PlovadDecoder()
    features = torch.randn(5, 512)
    conv_output = model(features)
    first, activation, second = model.classifier
    hidden = features @ first.weight[:, :, 0].T + first.bias
    expected = activation(hidden) @ second.weight[:, :, 0].T + second.bias
    torch.testing.assert_close(conv_output, expected.squeeze(1))

